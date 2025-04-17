from typing import Tuple

import torch.nn as nn
from torch import Tensor


class ResnetAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        input_channel: int,
        base_filters: int,
        activation: nn.Module,
    ):
        super().__init__()
        self.latent_dimension = latent_dimension

        # Build models
        self.encoder = _Encoder(input_channel, activation)
        self.enc_linear = nn.Linear(self.encoder.enc_out_dim, latent_dimension)

        self.dec_linear = nn.Linear(latent_dimension, self.encoder.enc_out_dim)
        self.decoder = _Decoder()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def encode(self, x: Tensor) -> Tensor:
        return self.enc_linear(self.encoder(x))

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(self.dec_linear(z))


class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """

    def __init__(self, c_in, c_out, k, s=1, p=1, mode="encode"):
        assert mode in ["encode", "decode"], "Mode must be either 'encode' or 'decode'."
        super().__init__()
        if mode == "encode":
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        elif mode == "decode":
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in

    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        relu = self.relu(conv1)
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        return self.relu(x + conv2)


class _Encoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channel, 8, kernel_size=3, padding=1, stride=1),  # 8 64 96
            nn.BatchNorm2d(8),
            activation(),
            ResBlock(8, 8, 3, 2, 1, "encode"),  # 8 32 48
            ResBlock(8, 16, 3, 2, 1, "encode"),  # 16 16 24
            ResBlock(16, 32, 3, 2, 1, "encode"),  # 32 8 12
            ResBlock(32, 32, 3, 2, 1, "encode"),  # 32 4 6
            nn.Flatten(),
        )
        self.enc_out_dim = 32 * 4 * 6

    def forward(self, x) -> Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(32, 32, 2, 2, 0, "decode"),  # 32 8 12
            ResBlock(32, 16, 2, 2, 0, "decode"),  # 16 16 24
            ResBlock(16, 16, 2, 2, 0, "decode"),  # 16 32 48
            ResBlock(16, 8, 2, 2, 0, "decode"),  # 8 64 96
            nn.ConvTranspose2d(8, 3, 3, 1, 1),  # 3 64 96
        )

    def forward(self, x) -> Tensor:
        x = x.reshape(x.shape[0], -1, 4, 6)
        return self.net(x)
