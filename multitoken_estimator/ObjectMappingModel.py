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
    squeeze_dim: int
    source_layer: int
    target_layer: int
    object_aggregation: Literal["mean", "first_token"]
    mapping_in: nn.Linear
    mapping_out: nn.Linear
    relation_prefixes: dict[str, str] | None

    def __init__(
        self,
        activations_dim: int,
        squeeze_dim: int,
        source_layer: int,
        target_layer: int,
        object_aggregation: Literal["mean", "first_token"],
        relation_prefixes: Optional[dict[str, str]] = None,
    ):
        super().__init__()
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.activations_dim = activations_dim
        self.squeeze_dim = squeeze_dim
        self.object_aggregation = object_aggregation
        self.relation_prefixes = relation_prefixes
        self.mapping_in = nn.Linear(activations_dim, squeeze_dim)
        self.mapping_out = nn.Linear(squeeze_dim, activations_dim)

    def forward(self, source_activation: torch.Tensor) -> torch.Tensor:
        return self.mapping_out(self.mapping_in(source_activation))

    def prefix_object(self, object_name: str, relation_name: str) -> str:
        return prefix_object(object_name, relation_name, self.relation_prefixes)

    def prefix_objects(
        self, object_names: Iterable[str], relation_name: str
    ) -> list[str]:
        return [
            self.prefix_object(object_name, relation_name)
            for object_name in object_names
        ]


def prefix_object(
    object_name: str, relation_name: str, relation_prefixes: dict[str, str] | None
) -> str:
    if relation_prefixes is None:
        return object_name
    prefix = relation_prefixes.get(relation_name, "").strip()
    return (prefix + " " + object_name).strip()
