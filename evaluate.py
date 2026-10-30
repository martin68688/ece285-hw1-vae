"""Inception Score and FID for a trained VAE, using PyTorch-Ignite as in
https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/

Generated images are compared with the 5,216 real training images. Images are
grayscale in [0, 1]; they are replicated to 3 channels and resized to 299x299
with the blog's PIL bilinear resize before being fed to Ignite's InceptionV3.

Example:
    python evaluate.py --run runs/vae_z128
"""
import argparse
import json
from pathlib import Path

import PIL.Image as Image
import torch
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image

from data import ChestXray
from train import get_device
from vae import VAE


def interpolate(batch):
    """The Ignite blog's PIL bilinear resize, after replicating grayscale to 3 channels."""
    arr = []
    for img in batch.repeat(1, 3, 1, 1):
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299, 299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)


def fid_is(fake, real, device, batch_size=128):
    """FID(fake, real) and IS(fake). `fake` and `real` are equal-length (N, 1, H, W) tensors.

    Ignite's default FID() (used in the blog) compares 1000-d InceptionV3 class
    probabilities; we also report FID on 2048-d pool3 features of the same
    torchvision ImageNet InceptionV3."""
    fid_metric = FID(device=device)
    fid2048_metric = FID(num_features=2048, feature_extractor=InceptionModel(return_features=True, device=device),
                         device=device)
    is_metric = InceptionScore(device=device, output_transform=lambda x: x[0])

    def evaluation_step(engine, batch):
        fake_b, real_b = batch
        return interpolate(fake_b).to(device), interpolate(real_b).to(device)

    evaluator = Engine(evaluation_step)
    fid_metric.attach(evaluator, "fid")
    fid2048_metric.attach(evaluator, "fid2048")
    is_metric.attach(evaluator, "is")
    loader = DataLoader(TensorDataset(fake, real), batch_size=batch_size)
    m = evaluator.run(loader, max_epochs=1).metrics
    return {"fid": m["fid"], "fid2048": m["fid2048"], "is": m["is"]}


@torch.no_grad()
def batched(fn, n, bs=256):
    return torch.cat([fn(min(bs, n - i), i) for i in range(0, n, bs)]).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--data", default="../data/chest_xray")
    p.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.8])
    p.add_argument("--real-baseline", action="store_true",
                   help="also score real-vs-real (two halves of the training set)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    run = Path(args.run)
    cfg = json.load(open(run / "config.json"))
    model = VAE(cfg["img_size"], 1, cfg["latent_dim"]).to(device)
    model.load_state_dict(torch.load(run / "model.pt", map_location=device))
    model.eval()

    real = ChestXray(args.data, "train", cfg["img_size"]).images.float() / 255.0
    n = len(real)
    results = {"n_images": n}

    for t in args.temperatures:
        fake = batched(lambda b, i: model.sample(b, device, temperature=t), n)
        results[f"sample_T{t}"] = fid_is(fake, real, device)
        save_image(fake[:64], run / f"generated_T{t}.png", nrow=8)
        print(f"samples T={t}: {results[f'sample_T{t}']}", flush=True)

    recon = batched(lambda b, i: torch.sigmoid(model(real[i:i + b].to(device))[0]), n)
    results["reconstruction"] = fid_is(recon, real, device)
    print(f"reconstructions: {results['reconstruction']}", flush=True)

    if args.real_baseline:
        perm = torch.randperm(n)
        half = n // 2
        results["real_vs_real"] = fid_is(real[perm[:half]], real[perm[half:2 * half]], device)
        results["real_vs_real"]["n_images"] = half
        save_image(real[perm[:64]], run / "real_examples.png", nrow=8)
        print(f"real vs real (N={half}): {results['real_vs_real']}", flush=True)

    json.dump(results, open(run / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
