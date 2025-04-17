from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        input_channel: int,
        base_filters: int,
        kernel_size: int,
        activation: nn.Module,
    ):
        super().__init__()
        self.example_input_array = torch.zeros(1, input_channel, 64, 96)
        self.latent_dimension = latent_dimension

        # Build models
        self.encoder = _Encoder(input_channel, base_filters, activation)
        self.decoder = _Decoder(input_channel, base_filters, activation)

        # distribution parameters
        self.fc_mu = nn.Linear(self.encoder.enc_out_dim, latent_dimension)
        self.fc_var = nn.Linear(self.encoder.enc_out_dim, latent_dimension)
        self.FC_dec = nn.Sequential(
            nn.Linear(latent_dimension, self.encoder.enc_out_dim), activation()
        )
        # for gaussian likelihood
        self.logscale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # encode
        mu, logvar = self.encode(x)
        # reparameterize
        std = torch.exp(logvar / 2)
        z = self.reparameterize(mu, std)
        # decode
        x_hat = self.decode(z)
        return x_hat, z, mu, logvar, std

    def encode(self, x: Tensor) -> Tensor:
        x_encoded = self.encoder(x)
        mu, logvar = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        x = self.FC_dec(z)
        return self.decoder(x)

    def reparameterize(self, mu: Tensor, std: Tensor) -> Tensor:
        eps = torch.finfo(std.dtype).eps
        q = torch.distributions.Normal(mu, std.clamp(min=eps))
        z = q.rsample()
        return z


class _Encoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        base_filters: int,
        activation: nn.Module,
    ) -> None:
        super().__init__()

        # Parameters
        hid = base_filters
        self.enc_out_dim = 8 * 12 * 4 * hid

        self.net = nn.Sequential(
            nn.Conv2d(input_channel, hid, kernel_size=5, padding=2, stride=2),
            activation(),
            nn.Conv2d(hid, hid, kernel_size=5, padding=2),
            activation(),
            nn.Conv2d(hid, 2 * hid, kernel_size=5, padding=2, stride=2),
            activation(),
            nn.Conv2d(2 * hid, 2 * hid, kernel_size=5, padding=2),
            activation(),
            nn.Conv2d(2 * hid, 4 * hid, kernel_size=5, padding=2, stride=2),
            activation(),
            nn.Flatten(),
        )

    def forward(self, x) -> Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        base_filters: int,
        activation: nn.Module,
    ) -> None:
        super().__init__()

        # Parameters
        hid = base_filters
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                4 * hid, 2 * hid, 5, padding=2, stride=2, output_padding=1
            ),
            activation(),
            nn.ConvTranspose2d(2 * hid, 2 * hid, 5, padding=2),
            activation(),
            nn.ConvTranspose2d(2 * hid, hid, 5, padding=2, stride=2, output_padding=1),
            activation(),
            nn.ConvTranspose2d(hid, hid, 5, padding=2),
            activation(),
            nn.ConvTranspose2d(
                hid, input_channel, 5, padding=2, stride=2, output_padding=1
            ),
        )

    def forward(self, x) -> Tensor:
        x = x.reshape(x.shape[0], -1, 8, 12)
        return self.net(x)
