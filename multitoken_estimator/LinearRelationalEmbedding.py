from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch

from multitoken_estimator.lib.torch_utils import low_rank_pinv


@dataclass
class InvertedLinearRelationalEmbedding:
    relation: str
    subject_layer: int
    object_layer: int
    weight_inverse: torch.Tensor
    bias: torch.Tensor
    rank: int
    object_aggregation: Literal["mean", "first_token"]
    metadata: dict[str, Any] | None = None

    @torch.no_grad()
    def calculate_subject_activation(
        self,
        object_activation: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        # match precision of weight_inverse and bias
        device = object_activation.device
        vec = (
            self.weight_inverse.to(device, dtype=torch.float32)
            @ (
                object_activation.float() - self.bias.to(device, dtype=torch.float32)
            ).t()
        ).mean(dim=1)
        if normalize:
            vec = vec / vec.norm()
        return vec


@dataclass
class LinearRelationalEmbedding:
    relation: str
    subject_layer: int
    object_layer: int
    weight: torch.Tensor
    bias: torch.Tensor
    object_aggregation: Literal["mean", "first_token"]
    metadata: dict[str, Any] | None = None

    def invert(
        self, rank: int, device: Optional[torch.device] = None
    ) -> InvertedLinearRelationalEmbedding:
        """Invert this LRE with a low-rank approximation, and optionally move to device."""
        if device is None:
            device = self.weight.device
        return InvertedLinearRelationalEmbedding(
            relation=self.relation,
            subject_layer=self.subject_layer,
            object_layer=self.object_layer,
            object_aggregation=self.object_aggregation,
            weight_inverse=low_rank_pinv(matrix=self.weight, rank=rank)
            .clone()
            .to(device),
            bias=self.bias.clone().to(device),
            rank=rank,
            metadata=self.metadata,
        )
