from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import torch.distributions
import torch.nn as nn
import torch.utils
from torch import Tensor

from koopman_rbc.models.components import Autoencoder, KoopmanOperator
from koopman_rbc.utils.metrics import (
    MeanSquaredScaledError,
    NormalizedSumSquaredError,
    NormSumSquaredScaledError,
)


class LRANModule(pl.LightningModule):
    def __init__(
        self,
        # Autoencoder params
        latent_dimension: int,
        input_channel: int,
        base_filters: int,
        kernel_size: int,
        ae_ckpt: str,
        # Loss params
        loss: str,
        horizon_weight: float,
        lambda_id: float,
        lambda_fwd: float,
        lambda_hid: float,
        lambda_reg: float,
        # Optimizer params
        lr_operator: float,
        lr_autoencoder: float,
        # Misc
        compile: bool = False,
        inv_transform: torch.nn.Module = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Model
        activation = nn.GELU
        self.autoencoder = Autoencoder(
            latent_dimension, input_channel, base_filters, kernel_size, activation
        )
        if ae_ckpt is not None:
            ckpt = torch.load(ae_ckpt, map_location=self.device)
            self.autoencoder.load_weights(ckpt, freeze=False)
        self.operator = KoopmanOperator(latent_dimension)

        # Loss
        if loss == "nssse":
            self.recon_loss = NormalizedSumSquaredError()
            self.fwd_loss = NormSumSquaredScaledError(horizon_weight=horizon_weight)
            self.hid_loss = NormSumSquaredScaledError(horizon_weight=horizon_weight)
        elif loss == "msse":
            self.recon_loss = torch.nn.MSELoss()
            self.fwd_loss = MeanSquaredScaledError(horizon_weight=horizon_weight)
            self.hid_loss = MeanSquaredScaledError(horizon_weight=horizon_weight)
        else:
            raise ValueError(f"Loss {loss} not supported")

        # Debugging
        self.example_input_array = torch.zeros(1, input_channel, 64, 96)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        g = self.autoencoder.encode(x)
        g_next = self.operator(g)
        x_hat = self.autoencoder.decode(g_next)
        return x_hat, g_next, g

    def model_step(self, x: Tensor, stage: str) -> Dict[str, Tensor]:
        seq_length = x.shape[1]
        g, g_hat, x_hat = [], [], []
        # Get ground truth for observables
        self.autoencoder.train(False)
        for tau in range(0, seq_length):
            g.append(self.autoencoder.encode(x[:, tau]))
        self.autoencoder.train(stage == "train")

        # Prediction
        g_hat.append(g[0])
        x_hat.append(self.autoencoder.decode(g[0]))
        # Predict sequence
        for tau in range(1, seq_length):
            g_hat.append(self.operator(g_hat[tau - 1]))
            x_hat.append(self.autoencoder.decode(g_hat[tau]))
        # To tensor
        g = torch.stack(g, dim=1)
        g_hat = torch.stack(g_hat, dim=1)
        x_hat = torch.stack(x_hat, dim=1)

        # Loss
        reconstruction = self.recon_loss(x_hat[:, :1], x[:, :1])
        forward = self.fwd_loss(x_hat[:, 1:], x[:, 1:])
        hidden = self.hid_loss(g_hat, g)
        reg = torch.zeros(1, device=self.device)
        loss = (
            self.hparams.lambda_id * reconstruction
            + self.hparams.lambda_fwd * forward
            + self.hparams.lambda_hid * hidden
            + self.hparams.lambda_reg * reg
        )

        # Log
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(
            {
                f"{stage}/loss/reconstruction": reconstruction,
                f"{stage}/loss/forward": forward,
                f"{stage}/loss/hidden": hidden,
                f"{stage}/loss/regularization": reg,
            },
            on_step=False,
            on_epoch=True,
        )

        # Apply inverse transform
        if self.hparams.inv_transform is not None:
            x = self.hparams.inv_transform(x.detach())
            x_hat = self.hparams.inv_transform(x_hat.detach())

        return {
            "loss": loss,
            "x": x,
            "x_hat": x_hat,
            "g": g,
            "g_hat": g_hat,
        }

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        return self.model_step(batch, stage="train")

    def validation_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        return self.model_step(batch, stage="val")

    def test_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        return self.model_step(batch, stage="test")

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.autoencoder = torch.compile(self.autoencoder)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            params=[
                {
                    "params": self.autoencoder.parameters(),
                    "lr": self.hparams.lr_autoencoder,
                },
                {"params": self.operator.parameters(), "lr": self.hparams.lr_operator},
            ],
        )
        return {"optimizer": optimizer}
