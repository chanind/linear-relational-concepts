from typing import Literal

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
