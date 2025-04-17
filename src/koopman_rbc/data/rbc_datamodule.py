from pathlib import Path
from typing import Dict, List

import lightning.pytorch as pl
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import v2

from .rbc_dataset import RBCDataset, RBCDatasetConfig, RBCType
from .test_dataset import TestDataset


class RBCDatamodule(pl.LightningDataModule):
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
        means: List[float] = [0.0, 0.0, 0.0],
        stds: List[float] = [1.0, 1.0, 1.0],
        include_control: bool = False,
        type: RBCType = RBCType.FULL,
        nr_episodes_train: int = 15,
        nr_episodes_val: int = 5,
        nr_episodes_test: int = 5,
        nr_test_sequences_per_episode: int = 10,
        val_for_test: bool = False,
    ) -> None:
        super().__init__()
        # DataModule parameters
        self.save_hyperparameters()
        # Dataset
        self.datasets: dict[str, Dataset] = {}
        self.configs: dict[str, Dict] = {}
        self.paths: dict[str, List[str]] = {}
        # Transform
        self.transform = v2.Normalize(
            mean=means,
            std=stds,
        )

    def prepare_data(self):
        # Get path to datasets
        dir = Path(self.hparams.data_dir) / f"ra{self.hparams.ra}"
        if not dir.exists():
            raise FileNotFoundError(f"Dataset not found at {dir}")
        self.files = sorted(dir.glob("*.h5"))

        # val + train
        self.paths["train"] = [
            str(f) for f in self.files[: self.hparams.nr_episodes_train]
        ]
        self.paths["val"] = [
            str(f)
            for f in self.files[
                self.hparams.nr_episodes_train : self.hparams.nr_episodes_train
                + self.hparams.nr_episodes_val
            ]
        ]
        self.configs["train"] = RBCDatasetConfig(
            type=self.hparams.type,
            dt=self.hparams.dt,
            start_index=0,
            end_index=self.hparams.train_length - 1,
            sequence_length=self.hparams.train_sequence_length,
        )
        self.configs["val"] = self.configs["train"]
        # test
        if not self.hparams.val_for_test:
            self.paths["test"] = [
                str(f)
                for f in self.files[
                    self.hparams.nr_episodes_train + self.hparams.nr_episodes_val :
                ]
            ]
        else:
            self.paths["test"] = self.paths["val"]

        self.configs["test"] = RBCDatasetConfig(
            type=self.hparams.type,
            dt=self.hparams.dt,
            start_index=0,
            end_index=self.hparams.test_length - 1,
            sequence_length=self.hparams.test_sequence_length,
        )

        # Read parameters from one of the datasets
        dset = RBCDataset(
            path=self.files[0],
            config=self.configs["train"],
        )
        self.dataset_params = dset.parameters

    def concat_datasets(self, stage: str):
        dsets = [
            RBCDataset(path=path, config=self.configs[stage], transform=self.transform)
            for path in self.paths[stage]
        ]

        # Wrap test set with less sequences
        if stage == "test":
            dsets = [
                TestDataset(
                    dset, nr_sequences=self.hparams.nr_test_sequences_per_episode
                )
                for dset in dsets
            ]

        return ConcatDataset(dsets)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.datasets["val"] = self.concat_datasets("val")
            self.datasets["train"] = self.concat_datasets("train")
        # Assign test dataset for use in dataloaders
        elif stage == "test":
            self.datasets["test"] = self.concat_datasets("test")
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
