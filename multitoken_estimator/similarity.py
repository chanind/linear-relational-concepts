from typing import Literal, Optional

import torch
from torch.nn.functional import cosine_similarity

from multitoken_estimator.MultitokenEstimator import ObjectActivation


def aggregate_similarity_at_layer(
    comparison: torch.Tensor,
    activations: list[ObjectActivation],
    layer: int,
    aggregation: Literal["mean", "weighted_mean"] = "mean",
) -> float:
    """
    Given a comparison matrix, activations, and a layer, return the
    mean similarity of the activations at the given layer.
    """
    layer_activations = [act.layer_activations[layer] for act in activations]
    similarity_tensors_stack = []
    for tok_activations in layer_activations:
        tok_similarities = [
            cosine_similarity(comparison, act, dim=0) for act in tok_activations
        ]
        if aggregation == "weighted_mean":
            similarity_tensors_stack.append(torch.stack(tok_similarities).mean(dim=0))
        elif aggregation == "mean":
            for similarity in tok_similarities:
                similarity_tensors_stack.append(similarity)
        else:
            raise ValueError(
                f"Unknown aggregation method: {aggregation}.  Must be one of 'mean', 'weighted_mean'."
            )
    return torch.stack(similarity_tensors_stack).mean().item()


def aggregate_similarity_at_all_layers(
    layer_comparisons: dict[int, torch.Tensor],
    activations: list[ObjectActivation],
    aggregation: Literal["mean", "weighted_mean"] = "mean",
) -> dict[int, float]:
    """
    Given a dictionary of layer comparisons, activations, and an aggregation
    method, return a dictionary of layer similarity values.
    """
    return {
        layer: aggregate_similarity_at_layer(
            comparison, activations, layer, aggregation=aggregation
        )
        for layer, comparison in layer_comparisons.items()
    }


def find_highest_similarity_layer(
    layer_comparisons: dict[int, torch.Tensor],
    activations: list[ObjectActivation],
    aggregation: Literal["mean", "weighted_mean"] = "mean",
    min_layer: int = 0,
    max_layer: Optional[int] = None,
) -> int:
    """
    Given a dictionary of layer comparisons, activations, and an aggregation
    method, return the layer with the highest similarity value.
    """
    if max_layer is None:
        max_layer = max(layer_comparisons.keys())
    similarities = aggregate_similarity_at_all_layers(
        layer_comparisons, activations, aggregation=aggregation
    )
    return max(
        range(min_layer, max_layer + 1),
        key=lambda layer: similarities[layer],
    )
