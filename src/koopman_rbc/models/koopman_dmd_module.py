from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import pykoopman as pk
import torch.distributions
import torch.nn as nn
import torch.utils
from pydmd import DMD
from torch import Tensor
from torchmetrics import Metric

from koopman_rbc.models.components import Autoencoder


class KoopmanDMDModule(pl.LightningModule):
    def __init__(
        self,
        autoencoder: Autoencoder,
        loss: Metric,
        action_dim: int,
        recursive: bool,
        include_control: bool,
        optimizer: torch.optim.Optimizer,
        lr_ae: float,
        lr_B: float,
        compile: bool,
        inv_transform: torch.nn.Module = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.autoencoder = autoencoder
        self.B = nn.Parameter(
            torch.randn(action_dim, autoencoder.latent_dimension, dtype=torch.float32)
        )
        self.loss = loss

        dmd = DMD(svd_rank=5)
        self.dmd_model = pk.Koopman(regressor=dmd)

        # Debugging
        self.example_input_array = (
            torch.zeros(32, 5, 1, 64, 96),
            torch.zeros(32, 5, action_dim),
        )

    def forward(self, x: Tensor, u: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        seq_length = x.shape[1]
        half = seq_length // 2

        # Encode sequence
        z = []
        for tau in range(0, seq_length):
            z.append(self.autoencoder.encode(x[:, tau]))
        z = torch.stack(z, dim=1)

        # Predict for each seq in batch separately
        z_hat = []
        for idx in z.size(0):
            # Fit model on each half sequence in the batch
            self.dmd_model.fit(z[idx, :half])
            # Predict the second half of the sequence
            z_hat.append(self.dmd_model.simulate(z[idx, 0], n_steps=seq_length - 1))
        z_hat = torch.concat(z_hat)

        # Decode sequence
        X_hat = []
        for tau in range(0, seq_length):
            X_hat.append(self.autoencoder.decode(z[:, tau]))
        X_hat = torch.stack(X_hat, dim=1)

        # Decode predicted sequence
        Y_hat = []
        for tau in range(0, seq_length - 1):
            Y_hat.append(self.autoencoder.decode(z_hat[:, tau]))
        Y_hat = torch.stack(Y_hat, dim=1)

        return Y_hat, X_hat

    def model_step(self, seq: Tensor, stage: str) -> Dict[str, Tensor]:
        if self.hparams.include_control:
            seq, u = seq
        else:
            u = None

        # Model forward
        y_hat, x_hat = self.forward(seq, u)

        # Loss
        seq_length = seq.shape[1]
        x = seq[:, : seq_length - 1]
        y = seq[:, 1:]
        x_loss = self.loss(x_hat, x)
        y_loss = self.loss(y_hat, y)
        loss = x_loss + (0.5 * y_loss)

        # Log
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/lossX", x_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}/lossY", y_loss, on_step=False, on_epoch=True, prog_bar=False)

        # Apply inverse transform
        if self.hparams.inv_transform is not None:
            x = self.hparams.inv_transform(x.detach())
            x_hat = self.hparams.inv_transform(x_hat.detach())
            y = self.hparams.inv_transform(y.detach())
            y_hat = self.hparams.inv_transform(y_hat.detach())

        return {
            "loss": loss,
            "x": x,
            "x_hat": x_hat,
            "y": y,
            "y_hat": y_hat,
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
        opt = self.hparams.optimizer(
            params=[
                {"params": self.autoencoder.parameters(), "lr_ae": self.hparams.lr_ae},
                {"params": self.B, "lr_B": self.hparams.lr_B},
            ],
        )
        return opt
