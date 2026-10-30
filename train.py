"""Train a VAE on the Kaggle chest X-ray dataset.

Example:
    python train.py --data ../data/chest_xray --out runs/vae_z128 --epochs 100
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import ChestXray
from vae import VAE, vae_loss


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device, beta):
    model.eval()
    tot = rec = kl = n = 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits, mu, logvar = model(x)
            l, r, k = vae_loss(logits, x, mu, logvar, beta)
            b = x.size(0)
            tot += l.item() * b; rec += r.item() * b; kl += k.item() * b; n += b
    return tot / n, rec / n, kl / n


def plot_curves(hist, path):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, key, title in zip(axes, ["loss", "recon", "kl"],
                              ["Negative ELBO", "Reconstruction (BCE)", "KL divergence"]):
        ax.plot(ep, hist[f"train_{key}"], label="train")
        ax.plot(ep, hist[f"test_{key}"], label="held-out")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../data/chest_xray")
    p.add_argument("--out", default="runs/vae")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    out = Path(args.out); (out / "samples").mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=2)

    train_set = ChestXray(args.data, "train", args.img_size)
    # The official "val" split has only 16 images, so the 624-image test split
    # is used as the held-out set for monitoring the loss.
    test_set = ChestXray(args.data, "test", args.img_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256)
    print(f"train={len(train_set)} test={len(test_set)} device={device}")

    model = VAE(args.img_size, 1, args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    fixed_z = torch.randn(64, args.latent_dim, device=device)
    fixed_x = torch.stack([test_set[i][0] for i in range(0, len(test_set), len(test_set) // 32)][:32]).to(device)
    hist = {f"{s}_{k}": [] for s in ["train", "test"] for k in ["loss", "recon", "kl"]}
    hist["iter_loss"] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot = rec = kl = n = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            logits, mu, logvar = model(x)
            loss, r, k = vae_loss(logits, x, mu, logvar, args.beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            b = x.size(0)
            tot += loss.item() * b; rec += r.item() * b; kl += k.item() * b; n += b
            hist["iter_loss"].append(loss.item())
        sched.step()
        te = evaluate(model, test_loader, device, args.beta)
        for key, v in zip(["loss", "recon", "kl"], [tot / n, rec / n, kl / n]):
            hist[f"train_{key}"].append(v)
        for key, v in zip(["loss", "recon", "kl"], te):
            hist[f"test_{key}"].append(v)
        print(f"epoch {epoch:3d} | train {tot/n:9.2f} (rec {rec/n:9.2f}, kl {kl/n:6.2f}) | "
              f"test {te[0]:9.2f} (rec {te[1]:9.2f}, kl {te[2]:6.2f}) | {time.time()-t0:.1f}s", flush=True)

        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                save_image(model.decode(fixed_z), out / "samples" / f"sample_ep{epoch:03d}.png", nrow=8)
                recon = torch.sigmoid(model(fixed_x)[0])
                save_image(torch.cat([fixed_x[:16], recon[:16]]), out / "samples" / f"recon_ep{epoch:03d}.png", nrow=16)
            torch.save(model.state_dict(), out / "model.pt")
            json.dump(hist, open(out / "history.json", "w"))
            plot_curves(hist, out / "loss_curves.png")


if __name__ == "__main__":
    main()
