from dataclasses import dataclass, replace
from typing import Any, Literal, Optional

import torch


@dataclass
class InvertedLinearRelationalEmbedding:
    relation: str
    subject_layer: int
    object_layer: int
    # store u, v, s, and bias separately to avoid storing the full weight matrix
    u: torch.Tensor
    s: torch.Tensor
    v: torch.Tensor
    bias: torch.Tensor
    rank: int
    object_aggregation: Literal["mean", "first_token"]
    metadata: dict[str, Any] | None = None

    def w_inv_times_vec(self, vec: torch.Tensor) -> torch.Tensor:
        device = vec.device
        dtype = vec.dtype
        s = self.s.to(device, dtype=dtype)
        v = self.v.to(device, dtype=dtype)
        u = self.u.to(device, dtype=dtype)
        # group u.T @ vec to avoid calculating larger matrices than needed
        return v @ torch.diag(1 / s) @ (u.T @ vec)

    @torch.no_grad()
    def calculate_subject_activation(
        self,
        object_activations: torch.Tensor,  # a tensor of shape (num_activations, hidden_activation_size)
        normalize: bool = True,
    ) -> torch.Tensor:
        # match precision of weight_inverse and bias
        device = object_activations.device
        dtype = object_activations.dtype
        bias = self.bias.to(device, dtype=dtype)
        unbiased_acts = object_activations - bias.unsqueeze(0)
        vec = self.w_inv_times_vec(unbiased_acts.T).mean(dim=1)

        if normalize:
            vec = vec / vec.norm()
        return vec

    def to(self, device: torch.device) -> "InvertedLinearRelationalEmbedding":
        return replace(
            self,
            u=self.u.to(device),
            v=self.v.to(device),
            s=self.s.to(device),
            bias=self.bias.to(device),
        )


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
        return InvertedLinearRelationalEmbedding(
            relation=self.relation,
            subject_layer=self.subject_layer,
            object_layer=self.object_layer,
            object_aggregation=self.object_aggregation,
            u=low_rank_u.detach().clone().to(device),
            s=low_rank_s.detach().clone().to(device),
            v=low_rank_v.detach().clone().to(device),
            bias=self.bias.detach().clone().to(device),
            rank=rank,
            metadata=self.metadata,
        )
