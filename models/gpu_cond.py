from jax.tree_util import tree_map


def gpu_cond(pred, true_func, false_func, args):
    return tree_map(
        lambda x, y: pred * x + (1 - pred) * y, true_func(args), false_func(args)
    )
