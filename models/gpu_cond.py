import jax


def gpu_cond(pred, true_func, false_func, args):
    return jax.tree_map(
        lambda x, y: pred * x + (1 - pred) * y, true_func(args), false_func(args)
    )
