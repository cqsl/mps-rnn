#!/usr/bin/env python3

import time
from functools import partial

import jax
import netket as nk
import numpy as np
import optax
from netket.jax import dtype_real
from netket.optimizer import identity_preconditioner
from netket.optimizer.qgt import QGTOnTheFly

from args import args
from ham import HeisenbergTriangular, Triangular
from models import MPS, MPSRNN1D, MPSRNN2D, TensorRNN2D, TensorRNNCmpr2D
from models.symmetry import symmetrize_spins
from readers import (
    convert_variables,
    try_load_enlarge,
    try_load_hierarchical,
    try_load_itensors,
)
from utils import init_out_dir, tree_size_real_nonzero, try_load_variables


def get_ham(*, _args=None):
    if not _args:
        _args = args

    if _args.boundary == "open":
        pbc = False
    elif _args.boundary == "peri":
        pbc = True
    else:
        raise ValueError(f"Unknown boundary: {_args.boundary}")

    if _args.ham.endswith("tri"):
        assert _args.ham_dim == 2
        if pbc and _args.sign == "mars":
            assert _args.L % 2 == 0
        graph = Triangular(_args.L, pbc)
    else:
        graph = nk.graph.Hypercube(length=_args.L, n_dim=_args.ham_dim, pbc=pbc)

    hilbert = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes)

    if _args.J == "afm":
        J = 1
    elif _args.J == "fm":
        J = -1
    else:
        raise ValueError(f"Unknown J: {_args.J}")

    if _args.ham == "ising":
        assert _args.sign == "none"
        H = nk.operator.IsingJax(hilbert=hilbert, graph=graph, J=J, h=_args.h)
    elif _args.ham.startswith("heis"):
        assert not _args.h
        if _args.ham.endswith("tri"):
            H = HeisenbergTriangular(
                hilbert=hilbert, graph=graph, J=J, sign_rule=_args.sign
            )
        else:
            H = nk.operator.Heisenberg(
                hilbert=hilbert,
                graph=graph,
                J=J,
                sign_rule=(_args.sign == "mars"),
            )
    else:
        raise ValueError(f"Unknown ham: {_args.ham}")

    return hilbert, H


def get_net(hilbert, *, _args=None):
    if not _args:
        _args = args

    net_args = dict(  # noqa: C408
        hilbert=hilbert,
        bond_dim=_args.bond_dim,
        zero_mag=_args.zero_mag,
        refl_sym=_args.refl_sym,
        affine=_args.affine,
        no_phase=_args.no_phase,
        no_w_phase=_args.no_w_phase,
        cond_psi=_args.cond_psi,
        reorder_type=_args.reorder_type,
        reorder_dim=_args.reorder_dim,
        dtype=_args.dtype,
    )

    if _args.net == "mps":
        assert _args.net_dim == 1
        Net = MPS
    elif _args.net == "mps_rnn":
        if _args.net_dim == 1:
            Net = MPSRNN1D
        elif _args.net_dim == 2:
            Net = MPSRNN2D
        else:
            raise ValueError(f"Unknown net_dim: {_args.net_dim}")
    elif _args.net == "tensor_rnn":
        assert _args.net_dim == 2
        Net = TensorRNN2D
    elif _args.net == "tensor_rnn_cmpr":
        assert _args.net_dim == 2
        Net = TensorRNNCmpr2D

    else:
        raise ValueError(f"Unknown net: {_args.net}")

    model = Net(**net_args)
    return model


def get_sampler(hilbert, *, _args=None):
    if not _args:
        _args = args

    return nk.sampler.ARDirectSampler(
        hilbert,
        dtype=dtype_real(_args.dtype),
        symmetrize_fun=symmetrize_spins if _args.refl_sym else None,
    )


def get_vstate(sampler, model, variables, *, _args=None, n_samples=None):
    if not _args:
        _args = args
    if not n_samples:
        n_samples = _args.batch_size

    return nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        chunk_size=_args.chunk_size,
        variables=variables,
        seed=_args.seed,
    )


def get_optimizer(*, _args=None):
    if not _args:
        _args = args

    if _args.optimizer.startswith("rk"):
        assert not _args.split_complex
        assert not _args.grad_clip
        assert not _args.train_only

        from netket import experimental as nkx

        if _args.optimizer == "rk12":
            Integrator = nkx.dynamics.RK12
        elif _args.optimizer == "rk23":
            Integrator = nkx.dynamics.RK23
        else:
            raise ValueError(f"Unknown optimizer: {_args.optimizer}")
        integrator = Integrator(dt=_args.lr, adaptive=True, rtol=1e-3, atol=1e-3)
        return integrator, None, None

    # Clip gradients after preconditioner
    chain = []
    if _args.grad_clip:
        chain.append(optax.clip_by_global_norm(_args.grad_clip))
    if _args.optimizer == "adam":
        chain.append(optax.scale_by_adam())
    lr = optax.linear_schedule(
        init_value=1e-6, end_value=args.lr, transition_steps=args.max_step // 10
    )
    chain.append(optax._src.alias._scale_by_learning_rate(lr))

    optimizer = optax.chain(*chain)

    if _args.train_only:
        names = _args.train_only.split(",")
        transforms = {True: optimizer, False: optax.set_to_zero()}

        def map_nested_fn(fn):
            def map_fn(d):
                return {
                    k: map_fn(v) if isinstance(v, dict) else fn(k, v)
                    for k, v in d.items()
                }

            return map_fn

        @map_nested_fn
        def label_fn(k, v):
            return k in names

        optimizer = optax.multi_transform(transforms, label_fn)

    if _args.split_complex:
        optimizer = optax.experimental.split_real_and_imaginary(optimizer)

    if _args.optimizer == "sr":
        solver = partial(jax.scipy.sparse.linalg.cg, tol=1e-7, atol=1e-7, maxiter=10)
        preconditioner = nk.optimizer.SR(
            qgt=QGTOnTheFly(), solver=solver, diag_shift=_args.diag_shift
        )
    else:
        assert not _args.diag_shift
        preconditioner = identity_preconditioner

    return optimizer, preconditioner


def get_vmc(H, vstate, optimizer, preconditioner, *, _args=None):
    if not _args:
        _args = args

    if _args.optimizer.startswith("rk"):
        assert preconditioner is None

        from netket import experimental as nkx

        solver = partial(jax.scipy.sparse.linalg.cg, tol=1e-7, atol=1e-7, maxiter=10)
        vmc = nkx.TDVP(
            H,
            variational_state=vstate,
            integrator=optimizer,
            propagation_type="imag",
            qgt=QGTOnTheFly(diag_shift=_args.diag_shift),
            linear_solver=solver,
            error_norm="qgt",
        )
    else:
        vmc = nk.VMC(
            H,
            variational_state=vstate,
            optimizer=optimizer,
            preconditioner=preconditioner,
        )

    logger = nk.logging.JsonLog(
        _args.log_filename,
        "w",
        save_params_every=_args.max_step // 100,
        write_every=_args.max_step // 100,
    )

    return vmc, logger


def try_load_variables_init(model, *, _args=None):
    if not _args:
        _args = args

    config = [
        ("init.mpack", try_load_variables),
        ("init_hi.mpack", try_load_hierarchical),
        ("init_el.mpack", try_load_enlarge),
        ("init.hdf5", try_load_itensors),
    ]

    for basename, func in config:
        filename = _args.full_out_dir + basename
        if func == try_load_variables:
            variables = func(filename)
        else:
            variables = func(filename, model, _args)
        if variables is not None:
            print(f"Found {filename}")
            variables = convert_variables(variables, _args)
            return variables

    print(f"Variables not found in {_args.full_out_dir}")
    return None


def try_load_variables_out(*, _args=None):
    if not _args:
        _args = args

    config = [
        ("out_ema.mpack", try_load_variables),
        ("out.mpack", try_load_variables),
    ]

    for basename, func in config:
        filename = _args.full_out_dir + basename
        variables = func(filename)
        if variables is not None:
            print(f"Found {filename}")
            variables = convert_variables(variables, _args)
            return variables

    print(f"Variables not found in {_args.full_out_dir}")
    return None


def main():
    init_out_dir()
    print(args.log_filename)

    hilbert, H = get_ham()

    model = get_net(hilbert)
    variables = try_load_variables_init(model)
    sampler = get_sampler(hilbert)
    vstate = get_vstate(sampler, model, variables)
    print("n_params", tree_size_real_nonzero(vstate.parameters))

    optimizer, preconditioner = get_optimizer()
    vmc, logger = get_vmc(H, vstate, optimizer, preconditioner)

    print("start_time", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start_time = time.time()
    if args.optimizer.startswith("rk"):
        t_max = args.lr * args.max_step
        vmc.run(
            T=t_max,
            out=logger,
            tstops=np.linspace(0, t_max, args.max_step + 1),
            show_progress=args.show_progress,
        )
    else:
        vmc.run(n_iter=args.max_step, out=logger, show_progress=args.show_progress)
    used_time = time.time() - start_time
    print("used_time", used_time)

    vstate.n_samples = args.estim_size
    energy = vstate.expect(H)
    print("energy", energy)


if __name__ == "__main__":
    main()
