from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, Literal, Optional

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
from multitoken_estimator.data_model import SampleDataModel
from multitoken_estimator.database import Database
from multitoken_estimator.lib.layer_matching import LayerMatcher
from multitoken_estimator.lib.logger import log_or_print, logger
from multitoken_estimator.lib.token_utils import find_prompt_answer_data
from multitoken_estimator.lib.torch_utils import get_device
from multitoken_estimator.lib.verify_answers_match_expected import (
    AnswerMatchResult,
    verify_answers_match_expected,
)
from multitoken_estimator.LinearRelationalEmbedding import (
    InvertedLinearRelationalEmbedding,
    LinearRelationalEmbedding,
)
from multitoken_estimator.MultitokenEstimator import (
    extract_object_activations_from_prompts,
)
from multitoken_estimator.PromptGenerator import Prompt, PromptGenerator
from multitoken_estimator.training.evaluate_relation_causality import (
    RelationCausalityResult,
    evaluate_relation_causality,
)

AggregationMethod = Literal["first", "mean"]


def evaluate_causality(
    model: nn.Module,
    tokenizer: Tokenizer,
    layer_matcher: LayerMatcher,
    concept_vector_layer: int,
    lres: list[LinearRelationalEmbedding],
    dataset: Database,
    batch_size: int = 32,
    inv_lre_rank: int = 100,
    object_tokens_aggregation: AggregationMethod = "mean",
    objects_aggregation: AggregationMethod = "mean",
    max_swaps_per_concept: int = 5,
    max_swaps_total: Optional[int] = None,
    magnitude_multiplier: float = 1.0,
    edit_single_layer_only: bool = False,
    use_remove_concept_projection_magnitude: bool = False,
    verbose: bool = True,
    use_zs_prompts: bool = True,
) -> list[RelationCausalityResult]:
    reformulated_dataset = _reformulate_zs_prompts(dataset, use_zs_prompts)
    prompt_generator = PromptGenerator(reformulated_dataset)
    relation_results: list[RelationCausalityResult] = []
    valid_relation_names = reformulated_dataset.get_relation_names()
    for lre in lres:
        if lre.relation not in valid_relation_names:
            logger.warning(
                f"Skipping relation {lre.relation} because it is not in the dataset"
            )
            continue
        relation_name = lre.relation
        objects_in_relation = reformulated_dataset.get_object_names_in_relation(
            relation_name
        )
        log_or_print(
            f"evaluating causality for concept group {relation_name} with {len(objects_in_relation)} concepts",
            verbose=verbose,
        )

        raw_prompts = prompt_generator.generate_prompts_for_relation(
            relation_name, num_fsl_examples=0, entity_modifiers=None
        )
        filter_prompts_result = _filter_incorrect_answer_prompts(
            model, tokenizer, raw_prompts, batch_size
        )
        valid_prompts = filter_prompts_result.valid_prompts
        prompt_answer_match_results = filter_prompts_result.prompt_answer_match_results
        valid_object_names = {prompt.object_name for prompt in valid_prompts}
        all_object_names = {prompt.object_name for prompt in raw_prompts}
        log_or_print(
            f"Model answers correctly {len(valid_prompts)} of {len(prompt_answer_match_results)} prompts.",
            verbose=verbose,
        )
        log_or_print(
            f"Valid objects: {len(valid_object_names)} of {len(all_object_names)}",
            verbose=verbose,
        )
        prompts_by_object = defaultdict(list)
        for prompt in valid_prompts:
            prompts_by_object[prompt.object_name].append(prompt)
        concepts = _build_concepts_from_lre(
            model=model,
            tokenizer=tokenizer,
            layer_matcher=layer_matcher,
            lre=lre,
            prompts_by_object=prompts_by_object,
            object_tokens_aggregation=object_tokens_aggregation,
            objects_aggregation=objects_aggregation,
            batch_size=batch_size,
            inv_lre_rank=inv_lre_rank,
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
            concept_vector_layer=concept_vector_layer,
            batch_size=batch_size,
            max_swaps_per_concept=max_swaps_per_concept,
            max_swaps_total=max_swaps_total,
            magnitude_multiplier=magnitude_multiplier,
            edit_single_layer_only=edit_single_layer_only,
            use_remove_concept_projection_magnitude=use_remove_concept_projection_magnitude,
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
    prompt_answer_match_results: list[AnswerMatchResult]
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
    lres: list[LinearRelationalEmbedding],
    dataset: Database,
    batch_size: int = 32,
    inv_lre_rank: int = 100,
    object_tokens_aggregation: AggregationMethod = "mean",
    objects_aggregation: AggregationMethod = "mean",
    verbose: bool = True,
    use_zs_prompts: bool = True,
) -> list[RelationAccuracyResult]:
    """
    Evaluate faithfulness of trained LRE dataset concepts
    """
    relation_results = []
    reformulated_dataset = _reformulate_zs_prompts(dataset, use_zs_prompts)
    prompt_generator = PromptGenerator(reformulated_dataset)
    valid_relation_names = reformulated_dataset.get_relation_names()
    for lre in lres:
        if lre.relation not in valid_relation_names:
            logger.warning(
                f"Skipping relation {lre.relation} because it is not in the dataset"
            )
            continue
        relation_name = lre.relation
        objects_in_relation = reformulated_dataset.get_object_names_in_relation(
            relation_name
        )
        log_or_print(
            f"evaluating faithfulness for concept group {relation_name} with {len(objects_in_relation)} concepts",
            verbose=verbose,
        )
        raw_prompts = prompt_generator.generate_prompts_for_relation(
            relation_name, num_fsl_examples=0, entity_modifiers=None
        )
        filter_prompts_result = _filter_incorrect_answer_prompts(
            model, tokenizer, raw_prompts, batch_size
        )
        valid_prompts = filter_prompts_result.valid_prompts
        prompt_answer_match_results = filter_prompts_result.prompt_answer_match_results
        valid_object_names = {prompt.object_name for prompt in valid_prompts}
        all_object_names = {prompt.object_name for prompt in raw_prompts}
        log_or_print(
            f"Model answers correctly {len(valid_prompts)} of {len(prompt_answer_match_results)} prompts.",
            verbose=verbose,
        )
        log_or_print(
            f"Valid objects: {len(valid_object_names)} of {len(all_object_names)}",
            verbose=verbose,
        )
        prompts_by_object = defaultdict(list)
        for prompt in valid_prompts:
            prompts_by_object[prompt.object_name].append(prompt)
        concepts = _build_concepts_from_lre(
            model=model,
            tokenizer=tokenizer,
            layer_matcher=layer_matcher,
            lre=lre,
            prompts_by_object=prompts_by_object,
            object_tokens_aggregation=object_tokens_aggregation,
            objects_aggregation=objects_aggregation,
            batch_size=batch_size,
            inv_lre_rank=inv_lre_rank,
        )
        matcher = ConceptMatcher(model, tokenizer, concepts, layer_matcher)

        valid_prompts = filter_prompts_result.valid_prompts
        prompt_answer_match_results = filter_prompts_result.prompt_answer_match_results
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
            prompt_answer_match_results=prompt_answer_match_results,
            prompt_eval_results=prompt_eval_results,
        )
        log_or_print(
            f"Relation {relation_name} accuracy: {relation_result.accuracy}",
            verbose=verbose,
        )
        relation_results.append(relation_result)
    return relation_results


def avg_faithfulness(
    faithfulness_results: Iterable[RelationAccuracyResult],
) -> float:
    faithfulness_sum: float = 0
    faithfulness_total = 0
    for res in faithfulness_results:
        # if no prompts were evaluated, skip this group
        if res.total > 0:
            faithfulness_total += 1
            faithfulness_sum += res.accuracy
    if faithfulness_total == 0:
        return 0
    return faithfulness_sum / faithfulness_total


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


@dataclass
class FilterPromptsResult:
    prompt_answer_match_results: list[AnswerMatchResult]
    valid_prompts: list[Prompt]


def _filter_incorrect_answer_prompts(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompts: Iterable[Prompt],
    batch_size: int,
) -> FilterPromptsResult:
    prompt_answer_match_results: list[AnswerMatchResult] = []
    expected_answers = []
    prompt_texts = []
    for prompt in prompts:
        prompt_texts.append(prompt.text)
        expected_answers.append(prompt.answer)
    prompt_answer_match_results = verify_answers_match_expected(
        model, tokenizer, prompt_texts, expected_answers, batch_size=batch_size
    )
    valid_prompts = [
        prompt
        for prompt, match_res in zip(prompts, prompt_answer_match_results)
        if match_res.answer_matches_expected
    ]
    return FilterPromptsResult(
        prompt_answer_match_results=prompt_answer_match_results,
        valid_prompts=valid_prompts,
    )


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


def _build_concepts_from_lre(
    model: nn.Module,
    tokenizer: Tokenizer,
    layer_matcher: LayerMatcher,
    lre: LinearRelationalEmbedding,
    prompts_by_object: dict[str, list[Prompt]],
    object_tokens_aggregation: AggregationMethod,
    objects_aggregation: AggregationMethod,
    inv_lre_rank: int,
    batch_size: int,
) -> list[Concept]:
    device = get_device(model)
    inverted_lre = lre.invert(rank=inv_lre_rank, device=device)
    concepts = []
    for object_name, prompts in prompts_by_object.items():
        concepts.append(
            _build_concept(
                model=model,
                tokenizer=tokenizer,
                layer_matcher=layer_matcher,
                object_name=object_name,
                inv_lre=inverted_lre,
                prompts=prompts,
                object_tokens_aggregation=object_tokens_aggregation,
                objects_aggregation=objects_aggregation,
                batch_size=batch_size,
            )
        )
    return concepts


def _build_concept(
    model: nn.Module,
    tokenizer: Tokenizer,
    layer_matcher: LayerMatcher,
    object_name: str,
    inv_lre: InvertedLinearRelationalEmbedding,
    prompts: list[Prompt],
    object_tokens_aggregation: AggregationMethod,
    objects_aggregation: AggregationMethod,
    batch_size: int,
    move_to_cpu: bool = True,
) -> Concept:
    object_layer = inv_lre.object_layer
    device = get_device(model)

    object_activations = extract_object_activations_from_prompts(
        model=model,
        tokenizer=tokenizer,
        layer_matcher=layer_matcher,
        object_name=object_name,
        object_prompts=prompts,
        batch_size=batch_size,
    )
    object_tensors = [
        _agg_token_acts(
            act.layer_activations[object_layer], method=object_tokens_aggregation
        ).to(device)
        for act in object_activations.activations
    ]
    concept_vec = (
        inv_lre.calculate_subject_activation(
            object_tensors, aggregation=objects_aggregation, normalize=True
        )
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
            "object_tokens_aggregation": object_tokens_aggregation,
            "objects_aggregation": objects_aggregation,
            "num_objects": len(object_tensors),
        },
    )


def _agg_token_acts(
    token_acts: tuple[torch.Tensor, ...], method: AggregationMethod
) -> torch.Tensor:
    if method == "first":
        return token_acts[0]
    elif method == "mean":
        return torch.mean(torch.stack(token_acts), dim=0)
    else:
        raise ValueError(f"Invalid token aggregation method {method}")
