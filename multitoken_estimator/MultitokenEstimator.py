import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from tokenizers import Tokenizer
from torch import nn

from multitoken_estimator.data_model import SampleDataModel
from multitoken_estimator.database import Database
from multitoken_estimator.lib.constants import DEFAULT_DEVICE
from multitoken_estimator.lib.extract_token_activations import extract_token_activations
from multitoken_estimator.lib.layer_matching import (
    LayerMatcher,
    collect_matching_layers,
)
from multitoken_estimator.lib.logger import log_or_print
from multitoken_estimator.lib.token_utils import find_prompt_answer_data
from multitoken_estimator.lib.verify_answers_match_expected import (
    verify_answers_match_expected,
)
from multitoken_estimator.PromptGenerator import (
    DEFAULT_ENTITY_MODIFIERS,
    EntityModifier,
    PromptGenerator,
)


@dataclass
class ObjectActivation:
    object: str
    object_tokens: list[int]
    base_prompt: str
    layer_activations: dict[int, tuple[torch.Tensor, ...]]


@dataclass
class ObjectActivations:
    object_name: str
    activations: list[ObjectActivation]


@dataclass
class ObjectRepresentation:
    object_name: str
    layer_representations: dict[int, torch.Tensor]
    activations: list[ObjectActivation]


class MultitokenEstimator:
    model: nn.Module
    tokenizer: Tokenizer
    device: torch.device
    database: Database
    prompt_generator: PromptGenerator
    layer_matcher: LayerMatcher

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        layer_matcher: LayerMatcher,
        database: Database,
        device: torch.device = DEFAULT_DEVICE,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.database = database
        self.prompt_generator = PromptGenerator(database)
        self.layer_matcher = layer_matcher

    def collect_object_activations(
        self,
        object: str,
        num_fsl_examples: int = 5,
        entity_modifiers: list[EntityModifier] = DEFAULT_ENTITY_MODIFIERS,
        exclude_fsl_examples_of_object: bool = True,
        batch_size: int = 8,
        max_prompts: Optional[int] = None,
        verbose: bool = True,
        valid_relations: Optional[set[str]] = None,
    ) -> ObjectActivations:
        self.model.eval()
        prompts = list(
            self.prompt_generator.generate_prompts_for_object(
                object,
                num_fsl_examples=num_fsl_examples,
                entity_modifiers=entity_modifiers,
                exclude_fsl_examples_of_object=exclude_fsl_examples_of_object,
                valid_relations=valid_relations,
            )
        )
        if max_prompts is not None:
            random.shuffle(prompts)
            prompts = prompts[:max_prompts]
        log_or_print(f"{len(prompts)} prompts generated for {object}", verbose=verbose)
        answer_match_results = verify_answers_match_expected(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=[prompt.text for prompt in prompts],
            expected_answers=[prompt.answer for prompt in prompts],
            batch_size=batch_size,
            show_progress=verbose,
        )
        matching_prompts = [
            prompt
            for prompt, match_result in zip(prompts, answer_match_results)
            if match_result.answer_matches_expected
        ]
        log_or_print(
            f"{len(matching_prompts)} of {len(prompts)} prompts matched expected answer",
            verbose=verbose,
        )
        prompt_answer_data = [
            find_prompt_answer_data(self.tokenizer, prompt.text, prompt.answer)
            for prompt in matching_prompts
        ]

        layers = collect_matching_layers(self.model, self.layer_matcher)
        layer_to_layer_num = {layer: i for i, layer in enumerate(layers)}
        log_or_print(
            f"extracting activations from {len(layers)} layers", verbose=verbose
        )
        raw_activations = extract_token_activations(
            self.model,
            self.tokenizer,
            layers=layers,
            texts=[prompt_answer.full_prompt for prompt_answer in prompt_answer_data],
            token_indices=[
                prompt_answer.output_answer_token_indices
                for prompt_answer in prompt_answer_data
            ],
            device=self.device,
            batch_size=batch_size,
            show_progress=verbose,
        )

        object_activations = []
        for prompt_answer, raw_activation in zip(prompt_answer_data, raw_activations):
            object_activations.append(
                ObjectActivation(
                    object=prompt_answer.answer,
                    object_tokens=prompt_answer.answer_tokens,
                    base_prompt=prompt_answer.base_prompt,
                    layer_activations={
                        layer_to_layer_num[layer]: tuple(activation)
                        for layer, activation in raw_activation.items()
                    },
                )
            )
        return ObjectActivations(object_name=object, activations=object_activations)

    def estimate_object_representation(
        self,
        object: str,
        num_fsl_examples: int = 5,
        entity_modifiers: list[EntityModifier] = DEFAULT_ENTITY_MODIFIERS,
        exclude_fsl_examples_of_object: bool = True,
        batch_size: int = 8,
        verbose: bool = True,
        estimation_method: Literal["mean", "weighted_mean", "first_token"] = "mean",
        max_prompts: Optional[int] = None,
        valid_relations: Optional[set[str]] = None,
    ) -> ObjectRepresentation:
        object_activations = self.collect_object_activations(
            object,
            num_fsl_examples=num_fsl_examples,
            entity_modifiers=entity_modifiers,
            exclude_fsl_examples_of_object=exclude_fsl_examples_of_object,
            batch_size=batch_size,
            max_prompts=max_prompts,
            verbose=verbose,
            valid_relations=valid_relations,
        )
        return self.estimate_object_representation_from_activations(
            object_activations, estimation_method=estimation_method
        )

    def estimate_object_representation_from_activations(
        self,
        object_activations: ObjectActivations,
        estimation_method: Literal["mean", "weighted_mean", "first_token"] = "mean",
    ) -> ObjectRepresentation:
        if estimation_method == "mean":
            return _estimate_object_representation_mean(object_activations)
        elif estimation_method == "weighted_mean":
            return _estimate_object_representation_weighted_mean(object_activations)
        elif estimation_method == "first_token":
            return _estimate_object_representation_first_token_mean(object_activations)
        else:
            raise ValueError(
                f"Unknown estimation method: {estimation_method}.  Must be one of 'mean', 'weighted_mean'."
            )

    def estimate_relation_object_representations(
        self,
        relation: str,
        num_fsl_examples: int = 5,
        entity_modifiers: list[EntityModifier] = DEFAULT_ENTITY_MODIFIERS,
        exclude_fsl_examples_of_object: bool = True,
        batch_size: int = 8,
        verbose: bool = True,
        estimation_method: Literal["mean", "weighted_mean", "first_token"] = "mean",
        max_prompts_per_object: Optional[int] = None,
        min_prompts_per_object: int = 0,
    ) -> list[ObjectRepresentation]:
        samples = self.database.query_all(
            SampleDataModel, lambda s: s.relation.name == relation
        )
        objects = {sample.object.name for sample in samples}
        object_representations = []
        for i, object in enumerate(objects):
            log_or_print(
                f"estimating representation for {object} ({i+1}/{len(objects)})",
                verbose=verbose,
            )
            if min_prompts_per_object > 0:
                num_prompts = len(
                    self.prompt_generator.generate_prompts_for_object(
                        object,
                        entity_modifiers=entity_modifiers,
                        exclude_fsl_examples_of_object=exclude_fsl_examples_of_object,
                        valid_relations={relation},
                    )
                )
                if num_prompts < min_prompts_per_object:
                    log_or_print(
                        f"not enough prompts for {object} ({num_prompts} < {min_prompts_per_object}), skipping",
                        verbose=verbose,
                    )
                    continue
            object_activations = self.collect_object_activations(
                object,
                num_fsl_examples=num_fsl_examples,
                entity_modifiers=entity_modifiers,
                exclude_fsl_examples_of_object=exclude_fsl_examples_of_object,
                batch_size=batch_size,
                max_prompts=max_prompts_per_object,
                verbose=verbose,
                valid_relations={relation},
            )
            if len(object_activations.activations) == 0:
                log_or_print(
                    f"no activations found for {object}, skipping", verbose=verbose
                )
                continue
            object_representations.append(
                self.estimate_object_representation_from_activations(
                    object_activations, estimation_method=estimation_method
                )
            )
        return object_representations


@torch.no_grad()
def _estimate_object_representation_mean(
    object_activations: ObjectActivations,
) -> ObjectRepresentation:
    all_activations_by_layer = defaultdict(list)
    for object_activation in object_activations.activations:
        for layer_num, activations in object_activation.layer_activations.items():
            for activation in activations:
                all_activations_by_layer[layer_num].append(activation)
    return ObjectRepresentation(
        object_name=object_activations.object_name,
        activations=object_activations.activations,
        layer_representations={
            layer_num: torch.stack(activations).mean(dim=0)
            for layer_num, activations in all_activations_by_layer.items()
        },
    )


@torch.no_grad()
def _estimate_object_representation_first_token_mean(
    object_activations: ObjectActivations,
) -> ObjectRepresentation:
    all_activations_by_layer = defaultdict(list)
    for object_activation in object_activations.activations:
        for layer_num, activations in object_activation.layer_activations.items():
            all_activations_by_layer[layer_num].append(activations[0])
    return ObjectRepresentation(
        object_name=object_activations.object_name,
        activations=object_activations.activations,
        layer_representations={
            layer_num: torch.stack(activations).mean(dim=0)
            for layer_num, activations in all_activations_by_layer.items()
        },
    )


@torch.no_grad()
def _estimate_object_representation_weighted_mean(
    object_activations: ObjectActivations,
) -> ObjectRepresentation:
    mean_activations_by_layer = defaultdict(list)
    for object_activation in object_activations.activations:
        for layer_num, activations in object_activation.layer_activations.items():
            mean_activations_by_layer[layer_num].append(
                torch.stack(activations).mean(dim=0)
            )

    return ObjectRepresentation(
        object_name=object_activations.object_name,
        activations=object_activations.activations,
        layer_representations={
            layer_num: torch.stack(activations).mean(dim=0)
            for layer_num, activations in mean_activations_by_layer.items()
        },
    )
