from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Concept:
    object: str
    relation: str
    layer: int
    vector: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f" {self.relation}: {self.object}"
