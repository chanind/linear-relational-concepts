from dataclasses import dataclass

import torch


@dataclass
class SubjectConceptDirection:
    relation: str
    object: str
    layer: int
    direction: torch.Tensor
