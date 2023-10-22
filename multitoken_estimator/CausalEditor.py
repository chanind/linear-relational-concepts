from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar, Union, cast

import torch
from tokenizers import Tokenizer
from torch import nn

from multitoken_estimator.Concept import Concept
from multitoken_estimator.lib.constants import DEFAULT_DEVICE
from multitoken_estimator.lib.layer_matching import (
    LayerMatcher,
    collect_matching_layers,
    get_layer_name,
)
from multitoken_estimator.lib.token_utils import (
    ensure_tokenizer_has_pad_token,
    find_final_word_token_index,
    make_inputs,
    predict_all_token_probs_from_input,
    predict_next_tokens_greedy,
)
from multitoken_estimator.lib.torch_utils import untuple_tensor
from multitoken_estimator.lib.TraceLayerDict import TraceLayerDict
from multitoken_estimator.lib.util import batchify

EditorSubject = Union[str, int, Callable[[str, list[int]], int]]


@dataclass
class ConceptSwapRequest:
    text: str
    subject: EditorSubject
    remove_concept: str
    add_concept: str


@dataclass
class ConceptSwapAndPredictGreedyRequest(ConceptSwapRequest):
    predict_num_tokens: int = 1


T = TypeVar("T")


class CausalEditor:
    concepts: list[Concept]
    model: nn.Module
    tokenizer: Tokenizer
    layers_matcher: LayerMatcher
    layer_name_to_num: dict[str, int]
    device: torch.device

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        concepts: list[Concept],
        layers_matcher: LayerMatcher,
        device: torch.device = DEFAULT_DEVICE,
    ) -> None:
        self.concepts = concepts
        self.model = model
        self.tokenizer = tokenizer
        self.layers_matcher = layers_matcher
        ensure_tokenizer_has_pad_token(tokenizer)
        num_layers = len(collect_matching_layers(self.model, self.layers_matcher))
        self.layer_name_to_num = {}
        for layer_num in range(num_layers):
            self.layer_name_to_num[
                get_layer_name(model, layers_matcher, layer_num)
            ] = layer_num
        self.device = device

    def swap_subject_concepts_and_predict_greedy(
        self,
        text: str,
        subject: EditorSubject,
        concept_vector_layer: int,
        remove_concept: str,
        add_concept: str,
        predict_num_tokens: int = 1,
        magnitude_multiplier: float = 1.0,
        # if False, edit the subject token at every layer
        edit_single_layer_only: bool = False,
        # if True, use the magnitude of the projection of the remove_concept against the subject's original activation
        # if False, use the magnitude of the subject's original activation
        use_remove_concept_projection_magnitude: bool = False,
    ) -> str:
        results = self.swap_subject_concepts_and_predict_greedy_bulk(
            [
                ConceptSwapAndPredictGreedyRequest(
                    text, subject, remove_concept, add_concept, predict_num_tokens
                )
            ],
            concept_vector_layer=concept_vector_layer,
            magnitude_multiplier=magnitude_multiplier,
            edit_single_layer_only=edit_single_layer_only,
            use_remove_concept_projection_magnitude=use_remove_concept_projection_magnitude,
        )
        return results[0]

    def swap_subject_concepts_and_predict_greedy_bulk(
        self,
        requests: Sequence[ConceptSwapAndPredictGreedyRequest],
        concept_vector_layer: int,
        magnitude_multiplier: float = 1.0,
        # if False, edit the subject token at every layer
        edit_single_layer_only: bool = False,
        # if True, use the magnitude of the projection of the remove_concept against the subject's original activation
        # if False, use the magnitude of the subject's original activation
        use_remove_concept_projection_magnitude: bool = False,
    ) -> list[str]:
        next_tokens = self.swap_subject_concepts_and_predict_tokens_greedy_bulk(
            requests,
            concept_vector_layer=concept_vector_layer,
            magnitude_multiplier=magnitude_multiplier,
            edit_single_layer_only=edit_single_layer_only,
            use_remove_concept_projection_magnitude=use_remove_concept_projection_magnitude,
        )
        return [self.tokenizer.decode(tokens) for tokens in next_tokens]

    def swap_subject_concepts_and_predict_tokens_greedy_bulk(
        self,
        requests: Sequence[ConceptSwapAndPredictGreedyRequest],
        concept_vector_layer: int,
        magnitude_multiplier: float = 1.0,
        # if False, edit the subject token at every layer
        edit_single_layer_only: bool = False,
        # if True, use the magnitude of the projection of the remove_concept against the subject's original activation
        # if False, use the magnitude of the subject's original activation
        use_remove_concept_projection_magnitude: bool = False,
        batch_size: int = 12,
        show_progress: bool = False,
    ) -> list[list[int]]:
        results: list[list[int]] = []
        for batch in batchify(requests, batch_size, show_progress=show_progress):

            def run_batch_fn() -> list[list[int]]:
                max_num_tokens = max(req.predict_num_tokens for req in batch)
                next_tokens = predict_next_tokens_greedy(
                    self.model,
                    self.tokenizer,
                    [req.text for req in batch],
                    num_tokens=max_num_tokens,
                    device=self.device,
                )
                return [
                    tokens[: req.predict_num_tokens]
                    for tokens, req in zip(next_tokens, batch)
                ]

            results.extend(
                self._swap_subject_concepts_and_run_batch(
                    batch,
                    run_fn=run_batch_fn,
                    concept_vector_layer=concept_vector_layer,
                    magnitude_multiplier=magnitude_multiplier,
                    edit_single_layer_only=edit_single_layer_only,
                    use_remove_concept_projection_magnitude=use_remove_concept_projection_magnitude,
                )
            )
        return results

    def swap_subject_concepts_and_predict_all_token_probs_bulk(
        self,
        requests: Sequence[ConceptSwapRequest],
        concept_vector_layer: int,
        magnitude_multiplier: float = 1.0,
        # if False, edit the subject token at every layer
        edit_single_layer_only: bool = False,
        # if True, use the magnitude of the projection of the remove_concept against the subject's original activation
        # if False, use the magnitude of the subject's original activation
        use_remove_concept_projection_magnitude: bool = False,
        batch_size: int = 12,
        show_progress: bool = False,
    ) -> list[torch.Tensor]:
        results: list[torch.Tensor] = []
        for batch in batchify(requests, batch_size, show_progress=show_progress):

            def run_batch_fn() -> list[torch.Tensor]:
                inputs = make_inputs(
                    self.tokenizer, [req.text for req in batch], device=self.device
                )
                return predict_all_token_probs_from_input(self.model, inputs)

            results.extend(
                self._swap_subject_concepts_and_run_batch(
                    batch,
                    run_fn=run_batch_fn,
                    concept_vector_layer=concept_vector_layer,
                    magnitude_multiplier=magnitude_multiplier,
                    edit_single_layer_only=edit_single_layer_only,
                    use_remove_concept_projection_magnitude=use_remove_concept_projection_magnitude,
                )
            )
        return results

    def _swap_subject_concepts_and_run_batch(
        self,
        requests: Sequence[ConceptSwapRequest],
        run_fn: Callable[[], T],
        concept_vector_layer: int,
        magnitude_multiplier: float = 1.0,
        # if False, edit the subject token at every layer
        edit_single_layer_only: bool = False,
        # if True, use the magnitude of the projection of the remove_concept against the subject's original activation
        # if False, use the magnitude of the subject's original activation
        use_remove_concept_projection_magnitude: bool = False,
    ) -> T:
        """
        Helper to run the given run_fn while swapping the subject concept for each request.
        The run_fn should run the model with the same batch of inputs as specified in the requests
        """
        subj_tokens = [self._find_subject_token(req) for req in requests]
        with torch.no_grad():
            remove_concept_vectors = [
                (
                    self._find_concept(req.remove_concept)
                    .vector.detach()
                    .clone()
                    .type(cast(torch.dtype, self.model.dtype))
                    .to(self.device)
                )
                for req in requests
            ]
            add_concept_vectors = [
                (
                    self._find_concept(req.add_concept)
                    .vector.detach()
                    .clone()
                    .type(cast(torch.dtype, self.model.dtype))
                    .to(self.device)
                )
                for req in requests
            ]

        def edit_model_output(output: torch.Tensor, layer_name: str) -> torch.Tensor:
            if (
                edit_single_layer_only
                and self.layer_name_to_num[layer_name] != concept_vector_layer
            ):
                return output
            fixed_output = untuple_tensor(output)
            for i, subj_token in enumerate(subj_tokens):
                remove_concept_vector = remove_concept_vectors[i]
                add_concept_vector = add_concept_vectors[i]
                original_subj_act = fixed_output[i][subj_token]
                if use_remove_concept_projection_magnitude:
                    base_magnitude = original_subj_act.dot(remove_concept_vector)
                else:
                    base_magnitude = original_subj_act.norm()
                magnitude = base_magnitude * magnitude_multiplier
                fixed_output[i][subj_token] = original_subj_act + magnitude * (
                    add_concept_vector - remove_concept_vector
                )
            return output

        with torch.no_grad(), TraceLayerDict(
            self.model,
            layers=self.layer_name_to_num.keys(),
            edit_output=edit_model_output,
        ):
            return run_fn()

    def _find_subject_token(self, query: ConceptSwapRequest) -> int:
        text = query.text
        subject = query.subject
        if isinstance(subject, int):
            return subject
        if isinstance(subject, str):
            return find_final_word_token_index(self.tokenizer, text, subject)
        if callable(subject):
            return subject(text, self.tokenizer.encode(text))
        raise ValueError(f"Unknown subject type: {type(subject)}")

    def _find_concept(self, concept_name: str) -> Concept:
        for concept in self.concepts:
            if concept.name == concept_name:
                return concept
        raise ValueError(f"Unknown concept: {concept_name}")
