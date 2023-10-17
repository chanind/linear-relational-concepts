from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import torch
from tokenizers import Tokenizer
from torch import nn

from multitoken_estimator.constants import DEFAULT_DEVICE
from multitoken_estimator.database import Database
from multitoken_estimator.extract_token_activations import extract_token_activations
from multitoken_estimator.layer_matching import LayerMatcher, collect_matching_layers
from multitoken_estimator.logger import log_or_print
from multitoken_estimator.PromptGenerator import (
    DEFAULT_ENTITY_MODIFIERS,
    EntityModifier,
    PromptGenerator,
)
from multitoken_estimator.token_utils import find_prompt_answer_data
from multitoken_estimator.verify_answers_match_expected import (
    verify_answers_match_expected,
)


@dataclass
class ObjectActivation:
    object: str
    object_tokens: list[int]
    layer_activations: dict[int, tuple[torch.Tensor, ...]]


@dataclass
class ObjectActivations:
    object_name: str
    activations: list[ObjectActivation]


@dataclass
class ObjectRepresentation:
    object_name: str
    layer_representations: dict[int, torch.Tensor]


class MultitokenEstimator:
    model: nn.Module
    tokenizer: Tokenizer
    device: torch.device
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
        self.prompt_generator = PromptGenerator(database)
        self.layer_matcher = layer_matcher

    def collect_object_activations(
        self,
        object: str,
        num_fsl_examples: int = 5,
        entity_modifiers: list[EntityModifier] = DEFAULT_ENTITY_MODIFIERS,
        exclude_fsl_examples_of_object: bool = True,
        batch_size: int = 8,
        verbose: bool = True,
    ) -> ObjectActivations:
        prompts = list(
            self.prompt_generator.generate_prompts_for_object(
                object,
                num_fsl_examples=num_fsl_examples,
                entity_modifiers=entity_modifiers,
                exclude_fsl_examples_of_object=exclude_fsl_examples_of_object,
            )
        )
        log_or_print(f"{len(prompts)} generated for {object}", verbose=verbose)
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
        estimation_method: Literal["mean", "weighted_mean"] = "mean",
    ) -> ObjectRepresentation:
        object_activations = self.collect_object_activations(
            object,
            num_fsl_examples=num_fsl_examples,
            entity_modifiers=entity_modifiers,
            exclude_fsl_examples_of_object=exclude_fsl_examples_of_object,
            batch_size=batch_size,
            verbose=verbose,
        )
        if estimation_method == "mean":
            return _estimate_object_representation_mean(object_activations)
        elif estimation_method == "weighted_mean":
            return _estimate_object_representation_weighted_mean(object_activations)
        else:
            raise ValueError(
                f"Unknown estimation method: {estimation_method}.  Must be one of 'mean', 'weighted_mean'."
            )


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
        layer_representations={
            layer_num: torch.stack(activations).mean(dim=0)
            for layer_num, activations in mean_activations_by_layer.items()
        },
    )
