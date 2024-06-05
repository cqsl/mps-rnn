#!/usr/bin/env python3

import numpy as np
import orjson
from matplotlib import pyplot as plt

ham_dir = "heis_tri_afm_mars_2d_L10"
net_dir = "chi16_zm_af_snake_sc_gc1"
out_filename = "./out/hi.pdf"
n_steps = 10**4

nets = [
    ("mps_rnn_1d", "1D MPS-RNN", "C1"),
    ("mps_rnn_2d", "2D MPS-RNN", "C2"),
    ("tensor_rnn_cmpr_2d", "Compressed tensor-RNN", "C4"),
    ("tensor_rnn_2d", "Tensor-RNN", "C3"),
]


# Moving average
def ma(data, *, window=10):
    out = np.nancumsum(data)
    out[window:] -= out[:-window]
    counts = np.cumsum(np.isfinite(data))
    counts[window:] -= counts[:-window]
    out /= np.maximum(counts, 1)
    return out


# Exponential moving average
def ema(data, *, momentum=0.9):
    data = np.asarray(data)
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, data.size):
        out[i] = momentum * out[i - 1] + (1 - momentum) * data[i]
    return out


def read_data(filename):
    print(filename)
    with open(filename, "rb") as f:
        data = f.read()
    data = orjson.loads(data)
    data = data["Energy"]["Mean"]["real"]
    data = ema(data)
    return data


def main():
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for i, (net_name, label, color) in enumerate(nets):
        data = read_data(f"./out/{ham_dir}/{net_name}_{net_dir}/out.log")
        ax.plot(range(i * n_steps, (i + 1) * n_steps), data, color=color, label=label)

    ax.set_xlabel("Step")
    ax.set_ylabel("Energy")
    ax.legend()
    ax.grid(alpha=0.5)

    fig.tight_layout()
    print(out_filename)
    fig.savefig(out_filename, bbox_inches="tight", pad_inches=0)
    plt.close()


if __name__ == "__main__":
    main()
