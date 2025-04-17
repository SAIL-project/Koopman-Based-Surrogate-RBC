import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from .rbc_dataset import RBCDataset


class TestDataset(Dataset[Tensor]):
    def __init__(
        self,
        dataset: RBCDataset,
        nr_sequences: int,
    ):
        self.dataset = dataset
        self.indices = np.random.randint(len(dataset), size=nr_sequences)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tensor:
        return self.dataset[self.indices[idx]]
