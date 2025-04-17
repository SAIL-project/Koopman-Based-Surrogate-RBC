from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import Metric

from koopman_rbc.models.components import Autoencoder


class KoopmanControlModule(pl.LightningModule):
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

        # DMD

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

        # Build Snapshot matrices
        zX = z[:, : seq_length - 1]
        zY = z[:, 1:]

        # Include control
        if self.hparams.include_control:
            u = u[:, : seq_length - 1]  # u only for zX
            B_batched = self.B.expand(z.size(0), self.B.size(0), self.B.size(1))
            zY = zY - torch.bmm(u, B_batched)

        # Solve for Koopman operator using first half of the sequence
        res = torch.linalg.lstsq(zX[:, :half], zY[:, :half])
        A = res.solution.to(torch.float32)

        # Compute predicted zY
        zY_hat = torch.matmul(zX, A)

        # Recursive prediction
        if self.hparams.recursive:
            # zY_hat = [Ag(x_1), A^2g(x_1),...,A^Tg(x_1)]
            zY_hat = []
            for tau in range(0, seq_length - 1):
                if tau == 0:
                    z_tau = zX[:, 0].unsqueeze(dim=1)
                else:
                    z_tau = zY_hat[tau - 1]
                # Evolve z using A
                z_next = torch.bmm(z_tau, A)
                # include control but why? TODO
                if self.hparams.include_control:
                    z_next += torch.bmm(u[:, tau].unsqueeze(dim=1), B_batched)
                zY_hat.append(z_next)

            zY_hat = torch.cat(zY_hat, dim=1)

        # Decode sequence
        X_hat = []
        for tau in range(0, seq_length - 1):
            X_hat.append(self.autoencoder.decode(zX[:, tau]))
        X_hat = torch.stack(X_hat, dim=1)

        # Decode predicted sequence
        Y_hat = []
        for tau in range(0, seq_length - 1):
            Y_hat.append(self.autoencoder.decode(zY_hat[:, tau]))
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
