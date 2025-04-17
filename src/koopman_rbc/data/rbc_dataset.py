import math
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class RBCType(StrEnum):
    CONVECTION = "convection"
    FULL = "full"


class RBCField(IntEnum):
    UY = 0
    UX = 1
    T = 2
    JCONV = 3


@dataclass
class RBCDatasetConfig:
    type: str = RBCType.FULL
    dt: float = 1.0
    start_index: int = 0
    end_index: int = 300
    sequence_length: int = 5
    include_control: bool = False


class RBCDataset(Dataset[Tensor]):
    def __init__(
        self,
        path: Path,
        config: RBCDatasetConfig,
        transform: torch.nn.Module | None = None,
    ):
        self.dataset = None
        self.path = path
        self.transform = transform

        # class parameters
        self.type = config.type
        self.dt = config.dt
        self.start_index = config.start_index
        self.end_index = config.end_index
        self.sequence_length = config.sequence_length
        self.include_control = config.include_control

        # dataset parameters
        try:
            with h5py.File(path, "r") as simulation:
                self.parameters = dict(simulation.attrs.items())

                # Compute step factor
                if "dt" in simulation.attrs:
                    self.step_factor = math.floor(self.dt / simulation.attrs["dt"])
                elif "action_duration" in simulation.attrs:
                    self.step_factor = math.floor(
                        self.dt / simulation.attrs["action_duration"]
                    )
                else:
                    raise ValueError("No dt or action_duration in dataset")

                # assertions
                assert self.step_factor > 0, (
                    "dataset dt must be a multiple of sim dt/agent_duration"
                )
                assert self.parameters["episode_length"] / self.dt >= self.end_index, (
                    "end_index out of bounds"
                )

        except Exception:
            raise ValueError(f"Error reading dataset: {path}")

    def __len__(self) -> int:
        return int(self.end_index - self.start_index - self.sequence_length + 2)

    def __getitem__(self, idx: int) -> Tensor:
        state_seq = torch.stack(
            [self.get_dataset_state(idx + j) for j in range(0, self.sequence_length)]
        )

        if self.include_control:
            control_seq = torch.stack(
                [
                    self.get_dataset_control(idx + j)
                    for j in range(0, self.sequence_length)
                ]
            )
            return state_seq, control_seq

        return state_seq

    def get_dataset_control(self, idx: int) -> Tensor:
        return torch.tensor(
            np.array(self.dataset["action"][idx * self.step_factor]),
            dtype=torch.float32,
        )

    def get_dataset_state(self, idx: int) -> Tensor:
        # Load singleton
        if self.dataset is None:
            self.dataset = h5py.File(self.path, "r")

        # Load state from dataset; multiply by step factor for correct dt
        # type: full
        state = torch.tensor(
            np.array(self.dataset["data"][idx * self.step_factor]), dtype=torch.float32
        )

        # type: convection field
        if self.type == RBCType.CONVECTION:
            state = (
                state[RBCField.UY] * (state[RBCField.T] - torch.mean(state[RBCField.T]))
            ).unsqueeze(0)

        # Apply transform
        if self.transform:
            state = self.transform(state)

        return state
