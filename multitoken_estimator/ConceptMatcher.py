from dataclasses import dataclass
from typing import Callable, Sequence, Union

import torch
from tokenizers import Tokenizer
from torch import nn

from multitoken_estimator.Concept import Concept
from multitoken_estimator.lib.constants import DEFAULT_DEVICE
from multitoken_estimator.lib.extract_token_activations import extract_token_activations
from multitoken_estimator.lib.layer_matching import (
    LayerMatcher,
    collect_matching_layers,
    get_layer_name,
)
from multitoken_estimator.lib.token_utils import (
    ensure_tokenizer_has_pad_token,
    find_final_subject_token_index,
)
from multitoken_estimator.lib.util import batchify

QuerySubject = Union[str, int, Callable[[str, list[int]], int]]


@dataclass
class ConceptMatchQuery:
    text: str
    subject: QuerySubject


@dataclass
class ConceptMatchResult:
    concept: str
    score: float


class ConceptMatcher:
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

    def query(self, query: str, subject: QuerySubject) -> dict[str, ConceptMatchResult]:
        return self.query_bulk([ConceptMatchQuery(query, subject)])[0]

    def query_bulk(
        self,
        queries: Sequence[ConceptMatchQuery],
        batch_size: int = 4,
        verbose: bool = False,
    ) -> list[dict[str, ConceptMatchResult]]:
        results: list[dict[str, ConceptMatchResult]] = []
        for batch in batchify(queries, batch_size, show_progress=verbose):
            results.extend(self._query_batch(batch))
        return results

    def _query_batch(
        self, queries: Sequence[ConceptMatchQuery]
    ) -> list[dict[str, ConceptMatchResult]]:
        subj_tokens = [self._find_subject_token(query) for query in queries]
        with torch.no_grad():
            batch_subj_token_activations = extract_token_activations(
                self.model,
                self.tokenizer,
                layers=self.layer_name_to_num.keys(),
                texts=[q.text for q in queries],
                token_indices=subj_tokens,
                device=self.device,
                # batching is handled already, so no nned to batch here too
                batch_size=len(queries),
                show_progress=False,
            )

        results: list[dict[str, ConceptMatchResult]] = []
        for raw_subj_token_activations in batch_subj_token_activations:
            concept_results: dict[str, ConceptMatchResult] = {}
            # need to replace the layer name with the layer number
            subj_token_activations = {
                self.layer_name_to_num[layer_name]: layer_activations[0]
                for layer_name, layer_activations in raw_subj_token_activations.items()
            }
            for concept in self.concepts:
                concept_results[concept.name] = _apply_concept_to_activations(
                    concept, subj_token_activations
                )
            results.append(concept_results)
        return results

    def _find_subject_token(self, query: ConceptMatchQuery) -> int:
        text = query.text
        subject = query.subject
        if isinstance(subject, int):
            return subject
        if isinstance(subject, str):
            return find_final_subject_token_index(self.tokenizer, text, subject)
        if callable(subject):
            return subject(text, self.tokenizer.encode(text))
        raise ValueError(f"Unknown subject type: {type(subject)}")


@torch.no_grad()
def _apply_concept_to_activations(
    concept: Concept, activations: dict[int, torch.Tensor]
) -> ConceptMatchResult:
    device = concept.vector.device
    score = torch.matmul(
        concept.vector,
        activations[concept.layer].to(device),
    ).item()
    return ConceptMatchResult(
        concept=concept.name,
        score=score,
    )