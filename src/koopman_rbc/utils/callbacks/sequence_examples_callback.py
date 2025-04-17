import logging
import pathlib
import tempfile

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger, WandbLogger
from matplotlib import animation
from matplotlib import pyplot as plt

from koopman_rbc.data import RBCField


class SequenceExamplesCallback(Callback):
    def __init__(self, train_freq: int = 2):
        self.train_freq = train_freq

        logger = logging.getLogger("matplotlib.animation")
        logger.setLevel(logging.ERROR)

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
            self.log_output(outputs, idx, "test", trainer.logger)

    def log_output(self, outputs: dict, idx: int, stage: str, logger: Logger):
        # unpack sequence
        x = outputs["x"][idx].cpu().numpy()
        x_hat = outputs["x_hat"][idx].cpu().numpy()

        # generate videos
        videos = []
        captions = []
        for field in [RBCField.T, RBCField.UX, RBCField.UY]:
            videos.append(self.sequence2video(x, "Ground Truth", field))
            captions.append(f"{field.name} - Ground Truth")
            videos.append(self.sequence2video(x_hat, "Prediction", field))
            captions.append(f"{field.name} - Prediction")
            videos.append(
                self.sequence2video(
                    np.abs(x - x_hat), "Difference", field, colormap="binary"
                )
            )
            captions.append(f"{field.name} - Difference")

        # log to wandb
        if isinstance(logger, WandbLogger):
            logger.log_video(f"{stage}/examples", videos, caption=captions)

    def sequence2video(
        self,
        sequence,
        caption: str,
        field=RBCField.T,
        colormap="coolwarm",
        fps=5,
    ) -> str:
        # set up path
        path = pathlib.Path(f"{tempfile.gettempdir()}/rbcmodel").resolve()
        path.mkdir(parents=True, exist_ok=True)
        # config fig
        fig, ax = plt.subplots()
        ax.set_axis_off()

        if colormap == "binary":
            vmin, vmax = None, None
        elif field == RBCField.T:
            vmin, vmax = 1, 2
        else:
            vmin, vmax = None, None

        # create video
        artists = []
        steps = sequence.shape[0]
        for i in range(steps):
            artists.append(
                [ax.imshow(sequence[i][field], cmap=colormap, vmin=vmin, vmax=vmax)],
            )
        ani = animation.ArtistAnimation(fig, artists, blit=True)

        # save as mp4
        writer = animation.FFMpegWriter(
            fps=fps, metadata=dict(artist="Me"), bitrate=1800
        )
        path = path / f"video_{field}_{caption}.mp4"
        ani.save(path, writer=writer)
        plt.close(fig)
        return str(path)
