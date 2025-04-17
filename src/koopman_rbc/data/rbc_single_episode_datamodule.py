from pathlib import Path
from typing import Dict, List

import lightning.pytorch as pl
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from .rbc_dataset import RBCDataset, RBCDatasetConfig, RBCType


class RBCSingleEpisodeDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_sequence_length: int = 5,
        test_sequence_length: int = 5,
        train_length: int = 400,
        test_length: int = 30,
        time_lag: int = 1,
        dt: float = 1.0,
        ra: float = 1e6,
        transform: nn.Module = None,
        include_control: bool = False,
        type: RBCType = RBCType.CONVECTION,
        episode_index: int = 0,
        train_val_split: List[float] = [0.9, 0.1],
    ) -> None:
        super().__init__()
        # DataModule parameters
        self.save_hyperparameters(logger=False)
        # Dataset
        self.datasets: dict[str, Dataset] = {}
        self.configs: dict[str, Dict] = {}

    def prepare_data(self):
        # Get path to datasets
        dir = Path(self.hparams.data_dir) / f"ra{self.hparams.ra}"
        if not dir.exists():
            raise FileNotFoundError(f"Dataset not found at {dir}")
        self.path = sorted(dir.glob("*.h5"))[self.hparams.episode_index]

        # Dataset parameters
        self.configs["fit"] = RBCDatasetConfig(
            type=self.hparams.type,
            dt=self.hparams.dt,
            start_idx=0,
            end_idx=self.hparams.train_length - 1,
            sequence_length=self.hparams.train_sequence_length,
            include_control=self.hparams.include_control,
        )
        self.configs["test"] = RBCDatasetConfig(
            type=self.hparams.type,
            dt=self.hparams.dt,
            start_idx=self.hparams.train_length,
            end_idx=self.hparams.train_length + self.hparams.test_length - 1,
            sequence_length=self.hparams.test_sequence_length,
            include_control=self.hparams.include_control,
        )

        # Read parameters from one of the datasets
        dset = RBCDataset(
            path=self.dataset_paths[0],
            cfg=self.configs["fit"],
        )
        self.dataset_params = dset.parameters

    def setup(self, stage: str):
        # Get datasets
        dset = RBCDataset(
            path=self.path,
            cfg=self.configs[stage],
            transform=self.hparams.transform,
        )

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train, val = random_split(dset, self.hparams.train_val_split)
            self.datasets["train"] = train
            self.datasets["val"] = val
        # Assign test dataset for use in dataloaders
        elif stage == "test":
            self.datasets["test"] = dset
        else:
            raise ValueError(f"Stage not implemented: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )
