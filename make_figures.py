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
from matplotlib.ticker import FuncFormatter
import numpy as np
from PIL import Image

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7})


LOSS_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "font.size": 6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
}
BLUE, BLUE2, GREY = "#0F4D92", "#3775BA", "#E4E4E4"
LATENT_COLORS = {128: BLUE, 64: "#42949E", 256: "#B64342"}


def loss_figure(runs, out):
    hist = {json.load(open(r / "config.json"))["latent_dim"]: json.load(open(r / "history.json")) for r in runs}
    main = json.load(open(runs[0] / "config.json"))["latent_dim"]
    h = hist[main]
    epochs = np.arange(1, len(h["train_loss"]) + 1)
    iters = np.arange(len(h["iter_loss"])) / (len(h["iter_loss"]) // len(h["train_loss"]))
    t = h["train_loss"]
    share = (t[0] - t[9]) / (t[0] - t[-1])
    short = FuncFormatter(lambda v, _: f"{v - 10000:.0f}")

    with plt.rc_context(LOSS_STYLE):
        fig = plt.figure(figsize=(5.5, 1.25))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1], left=0.07, right=0.985, top=0.875, bottom=0.235,
                              wspace=0.38, hspace=0.5)
        ax_a = fig.add_subplot(gs[:, 0])
        ax_b = fig.add_subplot(gs[:, 1])
        ax_c = fig.add_subplot(gs[0, 2])
        ax_d = fig.add_subplot(gs[1, 2], sharex=ax_c)

        ax_a.plot(iters, h["iter_loss"], color=GREY, lw=0.3, zorder=1)
        ax_a.plot(epochs, h["train_loss"], color=BLUE, lw=1.1, zorder=3)
        ax_a.plot(epochs, h["test_loss"], color=BLUE2, lw=0.8, ls=(0, (3, 1.5)), zorder=3)
        ax_a.axvline(10, color="#4D4D4D", lw=0.5, ls=":", zorder=2)
        ax_a.text(13, 10585, f"{share:.0%} of the drop\nby epoch 10", va="top", color="#272727")
        ax_a.text(99, 10345, "held-out", ha="right", va="bottom", color=BLUE2)
        ax_a.text(99, 10172, "train", ha="right", va="bottom", color=BLUE)
        ax_a.set(xlim=(0, 100), ylim=(10160, 10600), xlabel="Epoch", ylabel=r"Neg. ELBO $-\ 10^4$")
        ax_a.yaxis.set_major_formatter(short)

        for z in (64, 256, 128):
            ax_b.plot(epochs[4:], hist[z]["test_loss"][4:], color=LATENT_COLORS[z], lw=0.7)
            ax_d.plot(epochs, hist[z]["train_kl"], color=LATENT_COLORS[z], lw=0.7)
        for i, z in enumerate((64, 128, 256)):
            ax_b.text(0.98, 0.97 - i * 0.105, f"d = {z}", transform=ax_b.transAxes, ha="right", va="top",
                      color=LATENT_COLORS[z])
        ax_b.set(xlim=(5, 100), ylim=(10315, 10375), xlabel="Epoch", ylabel=r"Held-out $-\ 10^4$")
        ax_b.yaxis.set_major_formatter(short)

        ax_c.plot(epochs, h["train_recon"], color=BLUE, lw=0.9)
        ax_c.set(ylim=(10190, 10300), ylabel=r"BCE $-\ 10^4$")
        ax_c.yaxis.set_major_formatter(short)
        ax_c.tick_params(labelbottom=False)
        ax_d.set(xlim=(0, 100), ylim=(35, 75), xlabel="Epoch", ylabel="KL")

        for ax, label in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")):
            box = ax.get_position()
            fig.text(box.x0 - 0.055, box.y1 + 0.1, label, fontsize=8, fontweight="bold", va="top")
        fig.align_ylabels([ax_c, ax_d])
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
