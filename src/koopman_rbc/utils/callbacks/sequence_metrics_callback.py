import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torchmetrics import MeanSquaredError
import wandb

from koopman_rbc.utils.metrics import NormalizedSumError, NormalizedSumSquaredError


class SequenceMetric:
    def __init__(self, metric):
        self.metric = metric
        self.name = f"{metric.__class__.__name__}"
        self.data = []

    def update(
        self, prediction: list[Tensor], groundtruth: list[Tensor], batch_idx: int
    ):
        for sample_idx in range(groundtruth.shape[0]):
            for tau in range(groundtruth.shape[1]):
                self.data.append(
                    {
                        "idx": batch_idx * groundtruth.shape[0] + sample_idx,
                        "batch_idx": batch_idx,
                        "sample_idx": sample_idx,
                        "tau": tau,
                        "value": self.metric(
                            prediction[sample_idx][tau], groundtruth[sample_idx][tau]
                        ).item(),
                    }
                )

    def get_dataframe(self):
        return pd.DataFrame(self.data)


class SequenceMetricsCallback(Callback):
    def __init__(self, name: str, key_groundtruth: str, key_prediction: str):
        self.name = name
        self.key_groundtruth = key_groundtruth
        self.key_prediction = key_prediction

        # sequence metrics
        self.sequence_metrics = [
            SequenceMetric(metric=NormalizedSumSquaredError()),
            SequenceMetric(metric=NormalizedSumError()),
            SequenceMetric(metric=MeanSquaredError()),
        ]

    # Testing callbacks
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        for metric in self.sequence_metrics:
            metric.update(
                outputs[self.key_prediction].cpu(),
                outputs[self.key_groundtruth].cpu(),
                batch_idx,
            )

    def on_test_end(self, trainer, pl_module) -> None:
        for metric in self.sequence_metrics:
            df = metric.get_dataframe()
            im = self.plot_metrics(df, metric.name)
            if isinstance(trainer.logger, WandbLogger):
                trainer.logger.log_table(f"test/Table-{metric.name}", dataframe=df)
                trainer.logger.log_image(f"test/Plot-{metric.name}", [im])

    def plot_metrics(self, df: pd.DataFrame, metric: str):
        fig = plt.figure()
        sns.set_theme()
        ax = sns.lineplot(data=df, x="tau", y="value")
        ax.set_title(metric)
        ax.set_ylabel(metric)

        # save as image
        im = wandb.Image(fig, caption=metric)
        plt.close(fig)
        return im
