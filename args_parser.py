import argparse
import os
from datetime import datetime

import numpy as np
from jax import numpy as jnp

_parser_locked = False


def get_parser():
    if _parser_locked:
        return None

    parser = argparse.ArgumentParser(allow_abbrev=False)

    group = parser.add_argument_group("physics parameters")
    group.add_argument(
        "--ham",
        type=str,
        default="ising",
        choices=["ising", "heis", "heis_tri"],
        help="Hamiltonian type",
    )
    group.add_argument(
        "--J",
        type=str,
        default="afm",
        choices=["afm", "fm"],
        help="nearest neighbor interaction",
    )
    group.add_argument(
        "--boundary",
        type=str,
        default="open",
        choices=["open", "peri"],
        help="boundary condition",
    )
    group.add_argument(
        "--sign",
        type=str,
        default="none",
        choices=["none", "mars"],
        help="sign rule",
    )
    group.add_argument(
        "--ham_dim",
        type=int,
        default=2,
        choices=[1, 2],
        help="dimensions of the lattice",
    )
    group.add_argument(
        "--L",
        type=int,
        default=10,
        help="edge length of the lattice",
    )
    group.add_argument(
        "--h",
        type=float,
        default=0,
        help="transverse field",
    )

    group = parser.add_argument_group("network parameters")
    group.add_argument(
        "--net",
        type=str,
        default="mps",
        choices=["mps", "mps_rnn", "tensor_rnn", "tensor_rnn_cmpr"],
        help="network type",
    )
    group.add_argument(
        "--net_dim",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="dimensions of the network, 0 for matching `ham_dim`",
    )
    group.add_argument(
        "--bond_dim",
        type=int,
        default=2,
        help="bond dimension",
    )
    group.add_argument(
        "--zero_mag",
        action="store_true",
        help="apply zero magnetization constraint",
    )
    group.add_argument(
        "--refl_sym",
        action="store_true",
        help="apply reflectional symmetries",
    )
    group.add_argument(
        "--affine",
        action="store_true",
        help="use affine transformations",
    )
    group.add_argument(
        "--no_phase",
        action="store_true",
        help="fix phase = 0",
    )
    group.add_argument(
        "--no_w_phase",
        action="store_true",
        help="fix w = ones and c = 0 for the phase",
    )
    group.add_argument(
        "--cond_psi",
        action="store_true",
        help="use conditional wave functions",
    )
    group.add_argument(
        "--reorder_type",
        type=str,
        default="none",
        choices=["none", "snake"],
        help="type of the autoregressive order",
    )
    group.add_argument(
        "--reorder_dim",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="dimensions of the autoregressive order, 0 for matching `ham_dim`",
    )
    group.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64", "complex64", "complex128"],
        help="data type",
    )

    group = parser.add_argument_group("optimizer parameters")
    group.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed, 0 for randomized",
    )
    group.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "sr", "rk12", "rk23"],
        help="optimizer type",
    )
    group.add_argument(
        "--split_complex",
        action="store_true",
        help="split real and imaginary parts of parameters in the optimizer",
    )
    group.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size",
    )
    group.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate",
    )
    group.add_argument(
        "--diag_shift",
        type=float,
        default=0,
        help="diagonal shift of SR, 0 for matching `lr`",
    )
    group.add_argument(
        "--max_step",
        type=int,
        default=10**4,
        help="number of training/sampling steps",
    )
    group.add_argument(
        "--grad_clip",
        type=float,
        default=0,
        help="global norm to clip gradients, 0 for disabled",
    )
    group.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="chunk size, 0 for disabled",
    )
    group.add_argument(
        "--train_only",
        type=str,
        default="",
        help="names of parameters to train only, comma separated",
    )
    group.add_argument(
        "--estim_size",
        type=int,
        default=0,
        help="batch size to estimate the Hamiltonian, 0 for matching `batch_size`",
    )

    group = parser.add_argument_group("system parameters")
    group.add_argument(
        "--show_progress",
        action="store_true",
        help="show progress",
    )
    group.add_argument(
        "--cuda",
        type=str,
        default="0",
        help="GPU ID, empty string for disabled, multi-GPU is not supported yet",
    )
    group.add_argument(
        "--run_name",
        type=str,
        default="",
        help="output subdirectory to keep repeated runs, empty string for disabled",
    )
    group.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="./out",
        help="output directory, empty string for disabled",
    )

    return parser


def lock_parser():
    global _parser_locked
    _parser_locked = True


def get_ham_net_name(args):
    ham_name = "{ham}_{J}"
    if args.boundary != "open":
        ham_name += "_{boundary}"
    if args.sign != "none":
        ham_name += "_{sign}"
    ham_name += "_{ham_dim}d_L{L}"
    if args.h:
        ham_name += "_h{h:g}"
    ham_name = ham_name.format(**vars(args))

    net_name = "{net}_{net_dim}d_chi{bond_dim}"
    if args.zero_mag:
        net_name += "_zm"
    if args.refl_sym:
        net_name += "_rs"
    if args.affine:
        net_name += "_af"
    if args.no_phase:
        net_name += "_nop"
    if args.no_w_phase:
        net_name += "_now"
    if args.cond_psi:
        net_name += "_cp"
    if args.reorder_type != "none":
        net_name += "_{reorder_type}"
        if args.reorder_dim != args.ham_dim:
            net_name += "_{reorder_dim}d"

    if args.optimizer != "adam":
        net_name += "_{optimizer}"
    if args.split_complex:
        net_name += "_sc"
    if args.grad_clip:
        net_name += "_gc{grad_clip:g}"
    net_name = net_name.format(**vars(args))

    return ham_name, net_name


def post_init_args(args):
    if args.net_dim == 0:
        args.net_dim = args.ham_dim

    if args.reorder_dim == 0:
        args.reorder_dim = args.ham_dim

    if args.seed == 0:
        # The seed depends on the time and the PID
        args.seed = hash((datetime.now(), os.getpid())) & (2**32 - 1)

    if (
        args.optimizer == "sr" or args.optimizer.startswith("rk")
    ) and args.diag_shift == 0:
        args.diag_shift = args.lr

    if args.chunk_size == 0:
        args.chunk_size = None

    if args.estim_size == 0:
        args.estim_size = args.batch_size

    args.ham_name, args.net_name = get_ham_net_name(args)

    if args.dtype in ["float32", jnp.float32]:
        args.dtype = jnp.float32
    elif args.dtype in ["float64", jnp.float64]:
        args.dtype = jnp.float64
    elif args.dtype in ["complex64", jnp.complex64]:
        args.dtype = jnp.complex64
    elif args.dtype in ["complex128", jnp.complex128]:
        args.dtype = jnp.complex128
    else:
        raise ValueError(f"Unknown dtype: {args.dtype}")

    if args.out_dir:
        args.full_out_dir = "{out_dir}/{ham_name}/{net_name}/".format(**vars(args))
        if args.run_name:
            args.full_out_dir = "{full_out_dir}{run_name}/".format(**vars(args))
        args.log_filename = args.full_out_dir + "out"
    else:
        args.full_out_dir = None
        args.log_filename = None


def set_env(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    np.random.seed(args.seed)
