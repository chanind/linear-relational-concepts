from typing import Iterable, Literal, Optional

import torch
from torch import nn


class ObjectMappingModel(nn.Module):
    """
    Model for mapping an plain object representation like "France" to
    mimic the activation when it's an object at another layer, like
    "Paris is located in the country of France"
    """

    activations_dim: int
    target_rank: int
    source_layer: int
    target_layer: int
    object_aggregation: Literal["mean", "first_token"]
    mapping: nn.Linear
    prefix: str | None

    def __init__(
        self,
        activations_dim: int,
        target_rank: int,
        source_layer: int,
        target_layer: int,
        object_aggregation: Literal["mean", "first_token"],
        prefix: Optional[str] = None,
    ):
        super().__init__()
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.activations_dim = activations_dim
        self.target_rank = target_rank
        self.object_aggregation = object_aggregation
        self.prefix = prefix
        self.mapping = nn.Linear(activations_dim, target_rank)

    def forward(self, source_activation: torch.Tensor) -> torch.Tensor:
        return self.mapping(source_activation)

    def prefix_object(self, object_name: str) -> str:
        return prefix_object(object_name, self.prefix)

    def prefix_objects(self, object_names: Iterable[str]) -> list[str]:
        return [self.prefix_object(object_name) for object_name in object_names]


def prefix_object(object_name: str, prefix: str | None) -> str:
    if prefix is None:
        return object_name
    return (prefix.strip() + " " + object_name).strip()
