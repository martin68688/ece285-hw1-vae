"""Convolutional variational autoencoder implemented from scratch in PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def down(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, inplace=True),
    )


def up(cin, cout):
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, inplace=True),
    )


class VAE(nn.Module):
    """Encoder q(z|x) = N(mu, diag(sigma^2)), prior p(z) = N(0, I),
    decoder p(x|z) = Bernoulli(sigmoid(logits)) over pixel intensities."""

    def __init__(self, img_size=128, in_ch=1, latent_dim=128, channels=(32, 64, 128, 256, 512)):
        super().__init__()
        self.latent_dim = latent_dim
        n_down = len(channels)
        self.feat = img_size // 2 ** n_down  # spatial size of the bottleneck
        self.channels = channels

        enc, c = [], in_ch
        for ch in channels:
            enc.append(down(c, ch))
            c = ch
        self.encoder = nn.Sequential(*enc)
        flat = channels[-1] * self.feat * self.feat
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, flat)
        dec = []
        rev = list(channels[::-1])
        for cin, cout in zip(rev[:-1], rev[1:]):
            dec.append(up(cin, cout))
        self.decoder = nn.Sequential(*dec)
        self.out = nn.ConvTranspose2d(rev[-1], in_ch, 4, stride=2, padding=1)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode_logits(self, z):
        h = self.fc_dec(z).view(-1, self.channels[-1], self.feat, self.feat)
        return self.out(self.decoder(F.leaky_relu(h, 0.2)))

    def decode(self, z):
        return torch.sigmoid(self.decode_logits(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode_logits(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device, temperature=1.0):
        z = torch.randn(n, self.latent_dim, device=device) * temperature
        return self.decode(z)


def vae_loss(logits, x, mu, logvar, beta=1.0):
    """Negative ELBO, summed over pixels / latent dims and averaged over the batch."""
    b = x.size(0)
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / b
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
    return recon + beta * kl, recon, kl
