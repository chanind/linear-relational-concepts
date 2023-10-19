from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LinearRelationalEmbedding:
    relation: str
    subject_layer: int
    object_layer: int
    weight: torch.Tensor
    bias: torch.Tensor
    metadata: dict[str, Any] | None = None
