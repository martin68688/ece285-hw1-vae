"""Figures and the metrics table for the report.

Example:
    python make_figures.py --main runs/vae_z128 --ablation runs/vae_z64 runs/vae_z256 --out figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7})


def smooth(x, k=50):
    x = np.asarray(x)
    return np.convolve(x, np.ones(k) / k, mode="valid")


def loss_figure(runs, out):
    main = runs[0]
    h = json.load(open(main / "history.json"))
    iters_per_epoch = len(h["iter_loss"]) // len(h["train_loss"])
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.55))

    ax = axes[0]
    it = np.arange(len(h["iter_loss"])) / iters_per_epoch
    ax.plot(it, h["iter_loss"], color="0.8", lw=0.5, label="per iteration")
    ax.plot(it[49:], smooth(h["iter_loss"]), color="C0", lw=1, label="50-iteration average")
    ax.set_ylim(10150, 10800); ax.set_xlabel("epoch"); ax.set_title("(a) Training loss (neg. ELBO)")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    for run, c in zip(runs, ["C0", "C1", "C2"]):
        hh = json.load(open(run / "history.json"))
        z = json.load(open(run / "config.json"))["latent_dim"]
        ep = np.arange(1, len(hh["train_loss"]) + 1)
        ax.plot(ep, hh["train_loss"], color=c, lw=1, label=f"train, d={z}")
        ax.plot(ep, hh["test_loss"], color=c, lw=1, ls="--", label=f"held-out, d={z}")
    ax.set_ylim(10150, 10800); ax.set_xlabel("epoch"); ax.set_title("(b) Per-epoch neg. ELBO")
    colors = [Line2D([], [], color=c, lw=1, label=f"d={json.load(open(r / 'config.json'))['latent_dim']}")
              for r, c in zip(runs, ["C0", "C1", "C2"])]
    styles = [Line2D([], [], color="0.4", lw=1, label="train"),
              Line2D([], [], color="0.4", lw=1, ls="--", label="held-out")]
    ax.legend(handles=colors + styles, frameon=False, ncol=2, loc="upper right", columnspacing=1.0)

    ax = axes[2]
    ep = np.arange(1, len(h["train_kl"]) + 1)
    ax.plot(ep, h["train_recon"], color="C0", lw=1, label="reconstruction (BCE)")
    ax.set_ylim(10150, 10600)
    ax2 = ax.twinx()
    ax2.plot(ep, h["train_kl"], color="C3", lw=1, label="KL")
    ax2.set_ylim(0, 120)
    ax.set_xlabel("epoch"); ax.set_title("(c) Loss terms (train, d=128)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper right")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "loss_curves.pdf")
    plt.close(fig)


def sample_figure(main, out):
    panels = [("real_examples.png", "(a) Real training images"),
              ("generated_T1.0.png", "(b) Generated, T = 1.0"),
              ("generated_T0.8.png", "(c) Generated, T = 0.8")]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.8))
    for ax, (f, title) in zip(axes, panels):
        img = np.asarray(Image.open(main / f).convert("L"))
        # keep the top-left 3x4 block of the 8x8 grid (128 px tiles + 2 px padding)
        ax.imshow(img[: 3 * 130 + 2, : 4 * 130 + 2], cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, pad=2); ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.9, wspace=0.03)
    fig.savefig(out / "samples.pdf", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", required=True)
    p.add_argument("--ablation", nargs="*", default=[])
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    runs = [Path(args.main)] + [Path(r) for r in args.ablation]
    loss_figure(runs, out)
    sample_figure(Path(args.main), out)
    summary = {}
    for run in runs:
        z = json.load(open(run / "config.json"))["latent_dim"]
        h = json.load(open(run / "history.json"))
        summary[f"z{z}"] = {"metrics": json.load(open(run / "metrics.json")),
                            "final_train_loss": h["train_loss"][-1], "final_test_loss": h["test_loss"][-1],
                            "final_train_kl": h["train_kl"][-1], "final_test_recon": h["test_recon"][-1]}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
