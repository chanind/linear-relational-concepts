import pytest
import torch

from multitoken_estimator.MultitokenEstimator import ObjectActivation
from multitoken_estimator.similarity import aggregate_similarity_at_layer


def test_aggregate_similarity_at_layer_with_mean() -> None:
    activations = [
        ObjectActivation(
            object="CANADA",
            object_tokens=[1, 2],
            base_prompt="Oh,",
            layer_activations={
                2: (
                    torch.tensor([1.0, 2.0, 3.0]),
                    torch.tensor([0.0, 0.0, 0.0]),
                )
            },
        ),
        ObjectActivation(
            object="Canada",
            object_tokens=[1],
            base_prompt="Oh,",
            layer_activations={2: (torch.tensor([1.0, 2.0, 3.0]),)},
        ),
    ]
    assert aggregate_similarity_at_layer(
        torch.tensor([1.0, 2.0, 3.0]),
        activations,
        layer=2,
        aggregation="mean",
    ) == pytest.approx(2 / 3)


def test_aggregate_similarity_at_layer_with_weighted_mean() -> None:
    activations = [
        ObjectActivation(
            object="CANADA",
            object_tokens=[1, 2],
            base_prompt="Oh,",
            layer_activations={
                2: (
                    torch.tensor([1.0, 2.0, 3.0]),
                    torch.tensor([0.0, 0.0, 0.0]),
                )
            },
        ),
        ObjectActivation(
            object="Canada",
            object_tokens=[1],
            base_prompt="Oh,",
            layer_activations={2: (torch.tensor([1.0, 2.0, 3.0]),)},
        ),
    ]
    assert aggregate_similarity_at_layer(
        torch.tensor([1.0, 2.0, 3.0]),
        activations,
        layer=2,
        aggregation="weighted_mean",
    ) == pytest.approx(3 / 4)
