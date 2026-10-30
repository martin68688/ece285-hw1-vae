# VAE on chest X-rays (Homework 1)

Convolutional VAE written from scratch in PyTorch, trained on the Kaggle
[chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset at 128x128,
evaluated with PyTorch-Ignite's `FID` and `InceptionScore`.

| File | Purpose |
|---|---|
| `vae.py` | VAE model and negative-ELBO loss |
| `data.py` | Grayscale / resize / center-crop preprocessing, cached as uint8 tensors |
| `train.py` | Training loop, loss logging, loss curves, sample grids |
| `evaluate.py` | IS and FID with the Ignite blog's PIL resize (Ignite default 1000-d and 2048-d pool3) for samples, reconstructions, real-vs-real |
| `make_figures.py` | Figures and `summary.json` used in the report |

```bash
pip install -r requirements.txt
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia && unzip chest-xray-pneumonia.zip -d ../data
for z in 128 64 256; do python train.py --data ../data/chest_xray --latent-dim $z --epochs 100 --out runs/vae_z$z; done
python evaluate.py --run runs/vae_z128 --data ../data/chest_xray --real-baseline
for z in 64 256; do python evaluate.py --run runs/vae_z$z --data ../data/chest_xray --temperatures 1.0; done
python make_figures.py --main runs/vae_z128 --ablation runs/vae_z64 runs/vae_z256 --out figures
```

Preprocessing, training, evaluation and figures for the reported results all ran on one RTX 3090 machine (about 1.2 s per epoch). The logs are in `logs/` and the metrics and
training histories in `runs/*/metrics.json` and `runs/*/history.json`.
