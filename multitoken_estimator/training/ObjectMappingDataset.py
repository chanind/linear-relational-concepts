from typing import NamedTuple

import torch
from torch.utils.data import Dataset


class ObjectMappingSample(NamedTuple):
    object_name: str
    source_vector: torch.Tensor
    target_vector: torch.Tensor


class ObjectMappingBatch(NamedTuple):
    object_names: list[str]
    source_vectors: torch.Tensor
    target_vectors: torch.Tensor


class ObjectMappingDataset(Dataset[ObjectMappingSample]):
    samples: list[ObjectMappingSample]

    def __init__(self, samples: list[ObjectMappingSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ObjectMappingSample:
        return self.samples[idx]
