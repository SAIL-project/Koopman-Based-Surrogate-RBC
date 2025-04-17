from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch import Tensor

from koopman_rbc.models.components import Autoencoder
from koopman_rbc.utils.metrics import NormalizedSumSquaredError


class AutoencoderLitModule(LightningModule):
    def __init__(
        self,
        latent_dimension: int,
        input_channel: int,
        base_filters: int,
        kernel_size: int,
        lr: float,
        loss: str,
        compile: bool,
        inv_transform: torch.nn.Module = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        self.activation = nn.GELU
        self.autoencoder = Autoencoder(
            latent_dimension, input_channel, base_filters, kernel_size, self.activation
        )

        # Loss function
        if loss == "mse":
            self.criterion = torch.nn.MSELoss()
        elif loss == "l1":
            self.criterion = torch.nn.L1Loss()
        elif loss == "nsse":
            self.criterion = NormalizedSumSquaredError()
        else:
            raise ValueError(f"Loss {loss} not supported")

        # Debugging
        self.example_input_array = torch.zeros(1, input_channel, 64, 96)

    def forward(self, x: Tensor) -> Tensor:
        # forward
        x_hat, _ = self.autoencoder(x)
        return x_hat

    def model_step(self, batch: Tensor, stage: str) -> Tuple[Tensor, Tensor, Tensor]:
        # check input dimensions
        assert batch.shape[1] == 1, (
            "Expect sequence length of 1 for autoencoder training"
        )
        x = batch.squeeze(dim=1)

        # model forward
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, x)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Apply inverse transform (ez to do here)
        if self.hparams.inv_transform is not None:
            x = self.hparams.inv_transform(x.detach())
            x_hat = self.hparams.inv_transform(x_hat.detach())

        return {
            "loss": loss,
            "x_hat": x_hat,
            "x": x,
        }

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self.model_step(batch, stage="test")

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.autoencoder = torch.compile(self.autoencoder)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            params=self.trainer.model.parameters(), lr=self.hparams.lr
        )
        return {"optimizer": optimizer}
