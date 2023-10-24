from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence

import torch
from tokenizers import Tokenizer
from torch import nn

from multitoken_estimator.CausalEditor import CausalEditor
from multitoken_estimator.Concept import Concept
from multitoken_estimator.ConceptMatcher import (
    ConceptMatcher,
    ConceptMatchQuery,
    ConceptMatchResult,
)
from multitoken_estimator.data.data_model import SampleDataModel
from multitoken_estimator.data.database import Database
from multitoken_estimator.lib.extract_token_activations import (
    extract_final_token_activations,
)
from multitoken_estimator.lib.layer_matching import LayerMatcher, get_layer_name
from multitoken_estimator.lib.logger import log_or_print, logger
from multitoken_estimator.lib.PromptValidator import PromptValidator
from multitoken_estimator.lib.token_utils import find_prompt_answer_data
from multitoken_estimator.lib.torch_utils import get_device
from multitoken_estimator.LinearRelationalEmbedding import (
    InvertedLinearRelationalEmbedding,
)
from multitoken_estimator.ObjectMappingModel import ObjectMappingModel
from multitoken_estimator.PromptGenerator import Prompt, PromptGenerator
from multitoken_estimator.RelationalConceptEstimator import RelationalConceptEstimator
from multitoken_estimator.training.evaluate_relation_causality import (
    RelationCausalityResult,
    evaluate_relation_causality,
)


def evaluate_causality(
    model: nn.Module,
    tokenizer: Tokenizer,
    layer_matcher: LayerMatcher,
    estimators: Sequence[RelationalConceptEstimator],
    dataset: Database,
    batch_size: int = 32,
    max_swaps_per_concept: int = 5,
    max_swaps_total: Optional[int] = None,
    magnitude_multiplier: float = 1.0,
    edit_single_layer_only: bool = False,
    use_remove_concept_projection_magnitude: bool = False,
    verbose: bool = True,
    use_zs_prompts: bool = True,
    prompt_validator: Optional[PromptValidator] = None,
) -> list[RelationCausalityResult]:
    if not prompt_validator:
        prompt_validator = PromptValidator(model, tokenizer)
    reformulated_dataset = _reformulate_zs_prompts(dataset, use_zs_prompts)
    prompt_generator = PromptGenerator(reformulated_dataset)
    relation_results: list[RelationCausalityResult] = []
    valid_relation_names = reformulated_dataset.get_relation_names()
    for estimator in estimators:
        if estimator.relation not in valid_relation_names:
            logger.warning(
                f"Skipping relation {estimator.relation} because it is not in the dataset"
            )
            continue
        relation_name = estimator.relation
        objects_in_relation = reformulated_dataset.get_object_names_in_relation(
            relation_name
        )
        log_or_print(
            f"evaluating causality for relation {relation_name} with {len(objects_in_relation)} objects",
            verbose=verbose,
        )

        raw_prompts = prompt_generator.generate_prompts_for_relation(
            relation_name, num_fsl_examples=0, entity_modifiers=None
        )
        valid_prompts = prompt_validator.filter_prompts(
            raw_prompts, batch_size=batch_size
        )
        valid_object_names = {prompt.object_name for prompt in valid_prompts}
        all_object_names = {prompt.object_name for prompt in raw_prompts}
        log_or_print(
            f"Model answers correctly {len(valid_prompts)} of {len(raw_prompts)} prompts.",
            verbose=verbose,
        )
        log_or_print(
            f"Valid objects: {len(valid_object_names)} of {len(all_object_names)}",
            verbose=verbose,
        )
        prompts_by_object = defaultdict(list)
        for prompt in valid_prompts:
            prompts_by_object[prompt.object_name].append(prompt)
        concepts = _build_concepts_from_concept_estimator(
            model=model,
            tokenizer=tokenizer,
            layer_matcher=layer_matcher,
            estimator=estimator,
            object_names=list(objects_in_relation),
            batch_size=batch_size,
        )
        editor = CausalEditor(model, tokenizer, concepts, layer_matcher)
        concept_prompts: dict[str, list[Prompt]] = defaultdict(list)
        object_name_to_concept_name = {
            concept.object: concept.name for concept in concepts
        }
        for prompt in valid_prompts:
            concept_prompts[object_name_to_concept_name[prompt.object_name]].append(
                prompt
            )
        relation_result = evaluate_relation_causality(
            relation_name=relation_name,
            editor=editor,
            concept_prompts=concept_prompts,
            concept_vector_layer=estimator.inv_lre.subject_layer,
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
    estimators: Sequence[RelationalConceptEstimator],
    dataset: Database,
    batch_size: int = 32,
    verbose: bool = True,
    use_zs_prompts: bool = True,
    prompt_validator: Optional[PromptValidator] = None,
) -> list[RelationAccuracyResult]:
    """
    Evaluate accuracy of trained LRE dataset concepts
    """
    if not prompt_validator:
        prompt_validator = PromptValidator(model, tokenizer)
    relation_results = []
    reformulated_dataset = _reformulate_zs_prompts(dataset, use_zs_prompts)
    prompt_generator = PromptGenerator(reformulated_dataset)
    valid_relation_names = reformulated_dataset.get_relation_names()
    for estimator in estimators:
        if estimator.relation not in valid_relation_names:
            logger.warning(
                f"Skipping relation {estimator.relation} because it is not in the dataset"
            )
            continue
        relation_name = estimator.relation
        objects_in_relation = reformulated_dataset.get_object_names_in_relation(
            relation_name
        )
        log_or_print(
            f"evaluating accuracy for relation {relation_name} with {len(objects_in_relation)} objects",
            verbose=verbose,
        )
        raw_prompts = prompt_generator.generate_prompts_for_relation(
            relation_name, num_fsl_examples=0, entity_modifiers=None
        )
        valid_prompts = prompt_validator.filter_prompts(
            raw_prompts, batch_size=batch_size
        )
        valid_object_names = {prompt.object_name for prompt in valid_prompts}
        all_object_names = {prompt.object_name for prompt in raw_prompts}
        log_or_print(
            f"Model answers correctly {len(valid_prompts)} of {len(raw_prompts)} prompts.",
            verbose=verbose,
        )
        log_or_print(
            f"Valid objects: {len(valid_object_names)} of {len(all_object_names)}",
            verbose=verbose,
        )
        prompts_by_object = defaultdict(list)
        for prompt in valid_prompts:
            prompts_by_object[prompt.object_name].append(prompt)
        concepts = _build_concepts_from_concept_estimator(
            model=model,
            tokenizer=tokenizer,
            layer_matcher=layer_matcher,
            estimator=estimator,
            object_names=list(objects_in_relation),
            batch_size=batch_size,
        )
        matcher = ConceptMatcher(model, tokenizer, concepts, layer_matcher)

        object_name_to_concept_name = {
            concept.object: concept.name for concept in concepts
        }

        matcher_queries = []
        for prompt in valid_prompts:
            # only check prompt where the model gets the correct answer
            matcher_queries.append(ConceptMatchQuery(prompt.text, prompt.subject))

        concept_results = matcher.query_bulk(
            matcher_queries,
            batch_size=batch_size,
        )
        prompt_eval_results = [
            PromptAccuracyResult(
                correct_concept=object_name_to_concept_name[prompt.object_name],
                prompt=prompt,
                concept_scores=concept_result,
                num_answer_tokens=_get_prompt_num_answer_tokens(
                    tokenizer, prompt=prompt
                ),
            )
            for prompt, concept_result in zip(valid_prompts, concept_results)
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


def _reformulate_zs_prompts(dataset: Database, use_zs_prompts: bool) -> Database:
    """Return new concepts data replacing prompts with zs_prompts if use_zs_prompts is True"""
    if not use_zs_prompts:
        return dataset

    def reformulate_sample(sample: SampleDataModel) -> SampleDataModel:
        if sample.relation.zs_templates is None:
            raise ValueError(f"Missing zs_prompts in LRE sample data for {sample.id}")
        new_relation = replace(
            sample.relation,
            templates=sample.relation.zs_templates,
        )
        return replace(
            sample,
            relation=new_relation,
        )

    return dataset.reformulate_samples(reformulate_sample)


def _build_concepts_from_concept_estimator(
    model: nn.Module,
    tokenizer: Tokenizer,
    layer_matcher: LayerMatcher,
    estimator: RelationalConceptEstimator,
    object_names: Sequence[str],
    batch_size: int,
) -> list[Concept]:
    object_source_activations = _get_object_source_activations(
        model=model,
        tokenizer=tokenizer,
        layer_matcher=layer_matcher,
        object_mapping=estimator.object_mapping,
        object_names=object_names,
        batch_size=batch_size,
    )
    concepts = []
    for object_name in object_names:
        concepts.append(
            estimator.estimate_concept(
                object_name=object_name,
                object_source_activation=object_source_activations[object_name],
            )
        )
    return concepts


def _get_object_source_activations(
    model: nn.Module,
    tokenizer: Tokenizer,
    layer_matcher: LayerMatcher,
    object_mapping: ObjectMappingModel,
    object_names: Sequence[str],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    layer_name = get_layer_name(model, layer_matcher, object_mapping.source_layer)
    source_activations = extract_final_token_activations(
        model,
        tokenizer,
        layers=[layer_name],
        texts=object_mapping.prefix_objects(object_names),
        device=get_device(model),
        batch_size=batch_size,
        show_progress=False,
        move_results_to_cpu=True,
    )
    source_activations_by_object = {
        object_name: source_activation[layer_name]
        for object_name, source_activation in zip(object_names, source_activations)
    }
    return source_activations_by_object


def _build_concept(
    object_name: str,
    inv_lre: InvertedLinearRelationalEmbedding,
    object_activation: torch.Tensor,
    move_to_cpu: bool = True,
) -> Concept:
    concept_vec = (
        inv_lre.calculate_subject_activation(object_activation, normalize=True)
        .detach()
        .clone()
    )
    if move_to_cpu:
        concept_vec = concept_vec.cpu()

    return Concept(
        relation=inv_lre.relation,
        object=object_name,
        layer=inv_lre.subject_layer,
        vector=concept_vec,
        metadata={
            "num_objects": 1,
        },
    )
