from typing import Dict

from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torchmetrics import MeanSquaredError

from koopman_rbc.utils.metrics import NormalizedSumError, NormalizedSumSquaredError


class MetricsCallback(Callback):
    def __init__(self, name: str, key_groundtruth: str, key_prediction: str):
        self.name = name
        self.key_groundtruth = key_groundtruth
        self.key_prediction = key_prediction

        # metrics
        self.metrics = [
            NormalizedSumSquaredError(),
            NormalizedSumError(),
            MeanSquaredError(),
            # StructuralSimilarityIndexMeasure(),
        ]

    # Training callbacks
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.log_metrics(outputs, stage="train")

    # Validation callbacks
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self.log_metrics(outputs, stage="val")

    # Testing callbacks
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self.log_metrics(outputs, stage="test")

    # Helper function
    def log_metrics(self, output: Dict[str, Tensor], stage: str):
        for metric in self.metrics:
            self.log(
                f"{stage}/{self.name}-{metric.__class__.__name__}",
                metric(
                    output[self.key_prediction].cpu(),
                    output[self.key_groundtruth].cpu(),
                ),
                on_epoch=True,
            )
