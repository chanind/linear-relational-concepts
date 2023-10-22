from typing import Literal

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

    def __init__(
        self,
        activations_dim: int,
        squeeze_dim: int,
        source_layer: int,
        target_layer: int,
        object_aggregation: Literal["mean", "first_token"],
    ):
        super().__init__()
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.activations_dim = activations_dim
        self.squeeze_dim = squeeze_dim
        self.object_aggregation = object_aggregation
        self.mapping_in = nn.Linear(activations_dim, squeeze_dim)
        self.mapping_out = nn.Linear(squeeze_dim, activations_dim)

    def forward(self, source_activation: torch.Tensor) -> torch.Tensor:
        return self.mapping_out(self.mapping_in(source_activation))
