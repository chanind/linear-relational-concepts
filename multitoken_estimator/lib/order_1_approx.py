from typing import Literal

import torch
from tokenizers import Tokenizer
from torch import nn

from multitoken_estimator.lib.torch_utils import get_device, untuple_tensor
from multitoken_estimator.lib.TraceLayer import TraceLayer
from multitoken_estimator.lib.TraceLayerDict import TraceLayerDict

# Heavily based on https://github.com/evandez/relations/blob/main/src/functional.py


@torch.no_grad()
@torch.inference_mode(mode=False)
def order_1_approx(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    subject_layer: str,
    object_layer: str,
    subject_index: int,
    object_pred_indices: list[int],
    object_aggregation: Literal["mean", "first_token"] = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a first-order approximation of the LM between `subject` and `object`.

    Very simply, this computes the Jacobian of object with respect to subject, as well as
    object - (J * subject) to approximate the bias.

    Returns the weight and bias

    This is an adapted version of order_1_approx from https://github.com/evandez/relations/blob/main/src/functional.py
    """
    device = get_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Precompute everything up to the subject, if there is anything before it.
    past_key_values = None
    input_ids = inputs.input_ids
    _subject_index = subject_index
    if _subject_index > 0:
        outputs = model(input_ids=input_ids[:, :_subject_index], use_cache=True)
        past_key_values = outputs.past_key_values
        input_ids = input_ids[:, _subject_index:]
        _subject_index = 0
    use_cache = past_key_values is not None

    # Precompute initial h and z.
    with TraceLayerDict(model, layers=(subject_layer, object_layer)) as ret:
        outputs = model(
            input_ids=input_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
    subject_layer_output = ret[subject_layer].output
    assert subject_layer_output is not None  # keep mypy happy
    subject_activation = untuple_tensor(subject_layer_output)[0, _subject_index]
    object_activation = _extract_object_activation(
        ret[object_layer], object_pred_indices, object_aggregation
    )

    # Now compute J and b.
    def compute_object_from_subject(subject_activation: torch.Tensor) -> torch.Tensor:
        def insert_h(
            output: tuple[torch.Tensor, ...], layer: str
        ) -> tuple[torch.Tensor, ...]:
            hs = untuple_tensor(output)
            if layer != subject_layer:
                return output
            hs[0, _subject_index] = subject_activation
            return output

        with TraceLayerDict(
            model, (subject_layer, object_layer), edit_output=insert_h
        ) as ret:
            model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        return _extract_object_activation(
            ret[object_layer], object_pred_indices, object_aggregation
        )

    assert subject_activation is not None
    weight = torch.autograd.functional.jacobian(
        compute_object_from_subject, subject_activation, vectorize=True
    )
    bias = object_activation[None] - subject_activation[None].mm(weight.t())

    # NB(evan): Something about the jacobian computation causes a lot of memory
    # fragmentation, or some kind of memory leak. This seems to help.
    torch.cuda.empty_cache()

    return weight, bias


def _extract_object_activation(
    object_layer_trace: TraceLayer,
    object_pred_indices: list[int],
    object_aggregation: Literal["mean", "first_token"],
) -> torch.Tensor:
    object_layer_output = object_layer_trace.output
    assert object_layer_output is not None  # keep mypy happy
    object_activations = untuple_tensor(object_layer_output)[0, object_pred_indices]
    if object_aggregation == "mean":
        return object_activations.mean(dim=0)
    elif object_aggregation == "first_token":
        return object_activations[0]
