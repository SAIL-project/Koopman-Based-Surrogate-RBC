from typing import Dict, List

import numpy as np
from wandb import Image
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger, WandbLogger
from matplotlib import pyplot as plt

from koopman_rbc.data.rbc_dataset import RBCField
from koopman_rbc.utils.plot import plot_difference, plot_field, RBCFieldVisualizer


class ExamplesCallback(Callback):
    def __init__(self, train_freq: int = 1, test_freq: int = 1):
        self.vis = RBCFieldVisualizer(vmin=1.0, vmax=2.0)
        self.train_freq = train_freq
        self.test_freq = test_freq

    # Training callbacks
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0 and trainer.current_epoch % self.train_freq == 0:
            with torch.no_grad():
                self.log_output(outputs, 0, "train", trainer.logger)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0 and trainer.current_epoch % self.train_freq == 0:
            with torch.no_grad():
                self.log_output(outputs, 0, "val", trainer.logger)

    # Testing callbacks
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        for idx in range(len(batch)):
            if batch_idx % self.test_freq == 0:
                self.log_output(outputs, idx, "test", trainer.logger)

    # Helper functions
    def log_output(
        self, outputs: Dict, idx: int, stage: str, logger: Logger
    ) -> List[Image]:
        # unpack
        x = outputs["x"][idx].cpu().numpy()
        x_hat = outputs["x_hat"][idx].cpu().numpy()

        images = []
        for field in [RBCField.T, RBCField.UX, RBCField.UY]:
            fig_gt, _, _ = plot_field(x, field)
            fig_pred, _, _ = plot_field(x_hat, field)
            fig_diff, _, _ = plot_difference(np.abs(x[field] - x_hat[field]))
            images.append(Image(fig_gt, caption=f"Ground Truth - {field.name}"))
            images.append(Image(fig_pred, caption=f"Prediction - {field.name}"))
            images.append(Image(fig_diff, caption=f"Difference - {field.name}"))
            plt.close(fig_gt)
            plt.close(fig_pred)
            plt.close(fig_diff)

        # log images
        if isinstance(logger, WandbLogger):
            logger.log_image(f"{stage}/examples", images)
