from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch


@dataclass
class InvertedLinearRelationalEmbedding:
    relation: str
    subject_layer: int
    object_layer: int
    u: torch.Tensor
    w_inv_vs: torch.Tensor
    w_inv_bias: torch.Tensor
    rank: int
    object_aggregation: Literal["mean", "first_token"]
    metadata: dict[str, Any] | None = None

    def calc_low_rank_target_activation(
        self, object_activation: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.u.T.to(object_activation.device, dtype=object_activation.dtype)
            @ object_activation
        )

    @torch.no_grad()
    def calculate_subject_activation(
        self,
        target_activation: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        # match precision of weight_inverse and bias
        device = target_activation.device
        w_inv_vs = self.w_inv_vs.to(device, dtype=target_activation.dtype)
        w_inv_bias = self.w_inv_bias.to(device, dtype=target_activation.dtype)
        vec = w_inv_vs @ target_activation - w_inv_bias
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

    @torch.no_grad()
    def invert(
        self, rank: int, device: Optional[torch.device] = None
    ) -> InvertedLinearRelationalEmbedding:
        """Invert this LRE with a low-rank approximation, and optionally move to device."""
        if device is None:
            device = self.weight.device
        u, s, v = torch.svd(self.weight.float())
        low_rank_u: torch.Tensor = u[:, :rank].to(self.weight.dtype)
        low_rank_v: torch.Tensor = v[:, :rank].to(self.weight.dtype)
        low_rank_s: torch.Tensor = s[:rank].to(self.weight.dtype)
        w_inv_vs = low_rank_v @ torch.diag(1 / low_rank_s)
        w_inv_bias = w_inv_vs @ (low_rank_u.T @ self.bias)
        return InvertedLinearRelationalEmbedding(
            relation=self.relation,
            subject_layer=self.subject_layer,
            object_layer=self.object_layer,
            object_aggregation=self.object_aggregation,
            u=low_rank_u.detach().clone().to(device),
            w_inv_vs=w_inv_vs.detach().clone().to(device),
            w_inv_bias=w_inv_bias.detach().clone().to(device),
            rank=rank,
            metadata=self.metadata,
        )
