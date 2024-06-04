#!/usr/bin/env python3

from args import args
from utils import tree_size_real_nonzero
from vmc import get_ham, get_net, get_sampler, get_vstate, try_load_variables_init


def main():
    hilbert, H = get_ham()

    model = get_net(hilbert)
    variables = try_load_variables_init(model)
    sampler = get_sampler(hilbert)
    vstate = get_vstate(sampler, model, variables)
    print("n_params", tree_size_real_nonzero(vstate.parameters))

    vstate.n_samples = args.estim_size
    energy = vstate.expect(H)
    print("energy", energy)


if __name__ == "__main__":
    main()
