from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from linear_relational import (
    CausalEditor,
    Concept,
    ConceptMatcher,
    ConceptMatchQuery,
    ConceptMatchResult,
    Prompt,
)
from tokenizers import Tokenizer
from torch import nn

from linear_relational_concepts.lib.layer_matching import LayerMatcher
from linear_relational_concepts.lib.logger import log_or_print
from linear_relational_concepts.lib.PromptValidator import PromptValidator
from linear_relational_concepts.lib.token_utils import find_prompt_answer_data
from linear_relational_concepts.lib.util import group_items
from linear_relational_concepts.PromptGenerator import PromptGenerator

from .evaluate_relation_causality import (
    RelationCausalityResult,
    evaluate_relation_causality,
)


def evaluate_causality(
    model: nn.Module,
    tokenizer: Tokenizer,
    layer_matcher: LayerMatcher,
    concepts: Sequence[Concept],
    prompt_generator: PromptGenerator,
    batch_size: int = 32,
    max_swaps_per_concept: int = 5,
    max_swaps_total: Optional[int] = None,
    magnitude_multiplier: float = 1.0,
    edit_single_layer_only: bool = False,
    use_remove_concept_projection_magnitude: bool = False,
    verbose: bool = True,
    prompt_validator: Optional[PromptValidator] = None,
) -> list[RelationCausalityResult]:
    if not prompt_validator:
        prompt_validator = PromptValidator(model, tokenizer)
    relation_results: list[RelationCausalityResult] = []
    concepts_by_relation = group_items(concepts, lambda c: c.relation)

    for relation_name, concepts_in_relation in concepts_by_relation.items():
        log_or_print(
            f"evaluating causality for relation {relation_name} with {len(relation_name)} trained concepts",
            verbose=verbose,
        )

        object_name_to_concept_name = {
            concept.object: concept.name for concept in concepts_in_relation
        }
        valid_prompts = _generate_test_prompts(
            relation_name=relation_name,
            concepts=concepts_in_relation,
            prompt_generator=prompt_generator,
            prompt_validator=prompt_validator,
            batch_size=batch_size,
            verbose=verbose,
        )

        editor = CausalEditor(model, tokenizer, concepts_in_relation, layer_matcher)
        concept_prompts: dict[str, list[Prompt]] = defaultdict(list)
        for prompt in valid_prompts:
            concept_prompts[object_name_to_concept_name[prompt.object_name]].append(
                prompt
            )

        relation_result = evaluate_relation_causality(
            relation_name=relation_name,
            editor=editor,
            concept_prompts=concept_prompts,
            concept_vector_layer=_get_concepts_layer(concepts_in_relation),
            batch_size=batch_size,
            max_swaps_per_concept=max_swaps_per_concept,
            max_swaps_total=max_swaps_total,
            magnitude_multiplier=magnitude_multiplier,
            edit_single_layer_only=edit_single_layer_only,
            use_remove_concept_projection_magnitude=use_remove_concept_projection_magnitude,
            verbose=False,
        )
        log_or_print(
            f"Relation {relation_name} causality: {relation_result.causality:.3f}",
            verbose=verbose,
        )
        relation_results.append(relation_result)
    return relation_results


@dataclass
class PromptAccuracyResult:
    correct_concept: str
    prompt: Prompt
    concept_scores: dict[str, ConceptMatchResult]
    num_answer_tokens: int

    @property
    def predicted_concept(self) -> str:
        max_concept_result = max(self.concept_scores.values(), key=lambda c: c.score)
        return max_concept_result.concept

    @property
    def is_correct(self) -> bool:
        return self.correct_concept == self.predicted_concept


@dataclass
class RelationAccuracyResult:
    relation: str
    prompt_eval_results: list[PromptAccuracyResult]

    @property
    def total(self) -> int:
        return len(self.prompt_eval_results)

    @property
    def num_concepts_evaluated(self) -> int:
        return len(set(r.correct_concept for r in self.prompt_eval_results))

    @property
    def total_correct(self) -> int:
        return sum(1 for r in self.prompt_eval_results if r.is_correct)

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0
        return self.total_correct / self.total


def evaluate_relation_classification_accuracy(
    model: nn.Module,
    tokenizer: Tokenizer,
    layer_matcher: LayerMatcher,
    concepts: Sequence[Concept],
    prompt_generator: PromptGenerator,
    batch_size: int = 32,
    verbose: bool = True,
    prompt_validator: Optional[PromptValidator] = None,
) -> list[RelationAccuracyResult]:
    """
    Evaluate accuracy of trained LRE dataset concepts
    """
    if not prompt_validator:
        prompt_validator = PromptValidator(model, tokenizer)
    relation_results = []
    concepts_by_relation = group_items(concepts, lambda c: c.relation)
    for relation_name, concepts_in_relation in concepts_by_relation.items():
        log_or_print(
            f"evaluating accuracy for relation {relation_name} with {len(concepts_in_relation)} trained concepts",
            verbose=verbose,
        )
        object_name_to_concept_name = {
            concept.object: concept.name for concept in concepts_in_relation
        }
        valid_prompts = _generate_test_prompts(
            relation_name=relation_name,
            concepts=concepts_in_relation,
            prompt_generator=prompt_generator,
            prompt_validator=prompt_validator,
            batch_size=batch_size,
            verbose=verbose,
        )
        matcher = ConceptMatcher(model, tokenizer, concepts_in_relation, layer_matcher)

        matcher_queries = []
        for prompt in valid_prompts:
            # only check prompt where the model gets the correct answer
            matcher_queries.append(ConceptMatchQuery(prompt.text, prompt.subject))

        query_results = matcher.query_bulk(
            matcher_queries,
            batch_size=batch_size,
        )
        prompt_eval_results = [
            PromptAccuracyResult(
                correct_concept=object_name_to_concept_name[prompt.object_name],
                prompt=prompt,
                concept_scores=concept_result.concept_results,
                num_answer_tokens=_get_prompt_num_answer_tokens(
                    tokenizer, prompt=prompt
                ),
            )
            for prompt, concept_result in zip(valid_prompts, query_results)
        ]
        relation_result = RelationAccuracyResult(
            relation_name,
            prompt_eval_results=prompt_eval_results,
        )
        log_or_print(
            f"Relation {relation_name} accuracy: {relation_result.accuracy:.3f}",
            verbose=verbose,
        )
        relation_results.append(relation_result)
    return relation_results


def avg_accuracy(
    accuracy_results: Iterable[RelationAccuracyResult],
) -> float:
    accuracy_sum: float = 0
    accuracy_total = 0
    for res in accuracy_results:
        # if no prompts were evaluated, skip this group
        if res.total > 0:
            accuracy_total += 1
            accuracy_sum += res.accuracy
    if accuracy_total == 0:
        return 0
    return accuracy_sum / accuracy_total


def _get_prompt_num_answer_tokens(
    tokenizer: Tokenizer,
    prompt: Prompt,
) -> int:
    """Helper to return the number of answer tokens in a prompt"""
    answer_data = find_prompt_answer_data(
        tokenizer,
        base_prompt=prompt.text,
        answer=prompt.answer,
    )
    return answer_data.num_answer_tokens


def _get_concepts_layer(concepts: Sequence[Concept]) -> int:
    """Helper to get the layer of the first concept"""
    layer = concepts[0].layer
    for concept in concepts:
        if concept.layer != layer:
            raise ValueError(
                f"Concepts must all be from the same layer to evaluate, but got {layer} and {concept.layer}"
            )
    return layer


def _generate_test_prompts(
    relation_name: str,
    concepts: list[Concept],
    prompt_generator: PromptGenerator,
    prompt_validator: PromptValidator,
    batch_size: int,
    verbose: bool,
) -> list[Prompt]:
    testable_objects = {concept.object for concept in concepts}
    prompts = prompt_generator.generate_prompts_for_relation(
        relation_name, num_fsl_examples=0
    )
    testable_prompts = [
        prompt for prompt in prompts if prompt.object_name in testable_objects
    ]

    valid_prompts = prompt_validator.filter_prompts(
        testable_prompts, batch_size=batch_size, show_progress=verbose
    )
    log_or_print(
        f"Model answers correctly {len(valid_prompts)} of {len(prompts)} prompts.",
        verbose=verbose,
    )
    return valid_prompts
