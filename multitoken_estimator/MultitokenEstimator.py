import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from tokenizers import Tokenizer
from torch import nn

from multitoken_estimator.data.data_model import SampleDataModel
from multitoken_estimator.data.database import Database
from multitoken_estimator.lib.constants import DEFAULT_DEVICE
from multitoken_estimator.lib.extract_token_activations import extract_token_activations
from multitoken_estimator.lib.layer_matching import (
    LayerMatcher,
    collect_matching_layers,
)
from multitoken_estimator.lib.logger import log_or_print
from multitoken_estimator.lib.PromptValidator import PromptValidator
from multitoken_estimator.lib.token_utils import find_prompt_answer_data
from multitoken_estimator.lib.torch_utils import get_device
from multitoken_estimator.PromptGenerator import (
    DEFAULT_ENTITY_MODIFIERS,
    EntityModifier,
    Prompt,
    PromptGenerator,
)

EstimationMethod = Literal["mean", "weighted_mean", "first_token"]


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
    prompt_validator: PromptValidator

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        layer_matcher: LayerMatcher,
        database: Database,
        prompt_validator: Optional[PromptValidator] = None,
        device: torch.device = DEFAULT_DEVICE,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.database = database
        self.prompt_generator = PromptGenerator(database)
        self.layer_matcher = layer_matcher
        self.prompt_validator = prompt_validator or PromptValidator(model, tokenizer)

    def collect_object_activations(
        self,
        object_name: str,
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
                object_name,
                num_fsl_examples=num_fsl_examples,
                entity_modifiers=entity_modifiers,
                exclude_fsl_examples_of_object=exclude_fsl_examples_of_object,
                valid_relation_names=valid_relations,
            )
        )
        if max_prompts is not None:
            random.shuffle(prompts)
            prompts = prompts[:max_prompts]
        log_or_print(
            f"{len(prompts)} prompts generated for {object_name}", verbose=verbose
        )
        valid_prompts = self.prompt_validator.filter_prompts(
            prompts, batch_size=batch_size
        )
        log_or_print(
            f"{len(valid_prompts)} of {len(prompts)} prompts matched expected answer",
            verbose=verbose,
        )
        return extract_object_activations_from_prompts(
            model=self.model,
            tokenizer=self.tokenizer,
            layer_matcher=self.layer_matcher,
            object_name=object_name,
            object_prompts=valid_prompts,
            batch_size=batch_size,
            show_progress=verbose,
        )

    def estimate_object_representation(
        self,
        object_name: str,
        num_fsl_examples: int = 5,
        entity_modifiers: list[EntityModifier] = DEFAULT_ENTITY_MODIFIERS,
        exclude_fsl_examples_of_object: bool = True,
        batch_size: int = 8,
        verbose: bool = True,
        estimation_method: EstimationMethod = "mean",
        max_prompts: Optional[int] = None,
        valid_relations: Optional[set[str]] = None,
    ) -> ObjectRepresentation:
        object_activations = self.collect_object_activations(
            object_name,
            num_fsl_examples=num_fsl_examples,
            entity_modifiers=entity_modifiers,
            exclude_fsl_examples_of_object=exclude_fsl_examples_of_object,
            batch_size=batch_size,
            max_prompts=max_prompts,
            verbose=verbose,
            valid_relations=valid_relations,
        )
        return estimate_object_representation_from_activations(
            object_activations, estimation_method=estimation_method
        )

    def estimate_relation_object_representations(
        self,
        relation: str,
        num_fsl_examples: int = 5,
        entity_modifiers: list[EntityModifier] = DEFAULT_ENTITY_MODIFIERS,
        exclude_fsl_examples_of_object: bool = True,
        batch_size: int = 8,
        verbose: bool = True,
        estimation_method: EstimationMethod = "mean",
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
                        valid_relation_names={relation},
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
                estimate_object_representation_from_activations(
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


def extract_object_activations_from_prompts(
    model: nn.Module,
    tokenizer: Tokenizer,
    layer_matcher: LayerMatcher,
    object_name: str,
    object_prompts: list[Prompt],
    batch_size: int = 8,
    show_progress: bool = False,
) -> ObjectActivations:
    prompt_answer_data = [
        find_prompt_answer_data(tokenizer, prompt.text, prompt.answer)
        for prompt in object_prompts
    ]
    layers = collect_matching_layers(model, layer_matcher)
    layer_to_layer_num = {layer: i for i, layer in enumerate(layers)}
    raw_activations = extract_token_activations(
        model,
        tokenizer,
        layers=layers,
        texts=[prompt_answer.full_prompt for prompt_answer in prompt_answer_data],
        token_indices=[
            prompt_answer.output_answer_token_indices
            for prompt_answer in prompt_answer_data
        ],
        device=get_device(model),
        batch_size=batch_size,
        show_progress=show_progress,
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
    return ObjectActivations(object_name=object_name, activations=object_activations)


def estimate_object_representation_from_activations(
    object_activations: ObjectActivations,
    estimation_method: EstimationMethod = "mean",
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
