import os
from typing import get_args

import flax
import numpy as np
from jax.tree_util import tree_leaves, tree_map
from netket.utils.types import Array

from args import args


def ensure_dir(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass


def init_out_dir():
    if not args.full_out_dir:
        return
    ensure_dir(args.full_out_dir)


def leaf_size_real_nonzero(x):
    if not isinstance(x, get_args(Array)):
        return 0

    # If some but not all elements are exactly float zero, that means they are masked
    size = (x != 0).sum()
    if size == 0:
        size = x.size

    if np.iscomplexobj(x):
        size *= 2

    return size


def tree_size_real_nonzero(tree):
    return sum(tree_leaves(tree_map(leaf_size_real_nonzero, tree)))


def try_load_variables(filename):
    if not os.path.exists(filename):
        return None

    with open(filename, "rb") as f:
        data = f.read()

    variables = flax.serialization.msgpack_restore(data)
    return variables
