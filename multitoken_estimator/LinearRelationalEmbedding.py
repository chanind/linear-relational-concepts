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
    metadata: dict[str, Any] | None = None

    @torch.no_grad()
    def calculate_subject_activation(
        self,
        object_activations: list[torch.Tensor],
        aggregation: Literal["first", "mean"] = "mean",
        normalize: bool = True,
    ) -> torch.Tensor:
        if aggregation == "first":
            stacked_acts = torch.stack([object_activations[0]])
        elif aggregation == "mean":
            stacked_acts = torch.stack(object_activations)
        else:
            raise ValueError(f"Unknown aggregation {aggregation}")

        device = stacked_acts.device
        vec = (
            self.weight_inverse.to(device)
            @ (object_activations - self.bias.to(device)).t()
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
            weight_inverse=low_rank_pinv(matrix=self.weight, rank=rank)
            .clone()
            .to(device),
            bias=self.bias.clone().to(device),
            rank=rank,
            metadata=self.metadata,
        )
