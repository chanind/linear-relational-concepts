import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import torch
from tokenizers import Tokenizer
from torch import nn

from linear_relational_concepts.data.RelationDataset import RelationDataset
from linear_relational_concepts.lib.constants import DEFAULT_DEVICE
from linear_relational_concepts.lib.extract_token_activations import (
    extract_token_activations,
)
from linear_relational_concepts.lib.layer_matching import (
    LayerMatcher,
    collect_matching_layers,
)
from linear_relational_concepts.lib.logger import log_or_print
from linear_relational_concepts.lib.PromptValidator import PromptValidator
from linear_relational_concepts.lib.token_utils import find_prompt_answer_data
from linear_relational_concepts.lib.torch_utils import get_device
from linear_relational_concepts.PromptGenerator import (
    AUGMENTATION_MODIFIERS,
    EntityModifier,
    Prompt,
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


RepresentationEstimator = Callable[[Sequence[tuple[torch.Tensor, ...]]], torch.Tensor]
EstimationMethod = (
    Literal["mean", "weighted_mean", "first_token"] | RepresentationEstimator
)


def mean_estimator(
    object_activations: Sequence[tuple[torch.Tensor, ...]],
) -> torch.Tensor:
    all_activations: list[torch.Tensor] = []
    for token_activations in object_activations:
        all_activations.extend(token_activations)
    return torch.stack(all_activations).mean(dim=0)


def weighted_mean_estimator(
    object_activations: Sequence[tuple[torch.Tensor, ...]],
) -> torch.Tensor:
    grouped_activations: list[torch.Tensor] = []
    for token_activations in object_activations:
        grouped_activations.append(torch.stack(token_activations).mean(dim=0))
    return torch.stack(grouped_activations).mean(dim=0)


def first_token_estimator(
    object_activations: Sequence[tuple[torch.Tensor, ...]],
) -> torch.Tensor:
    all_activations: list[torch.Tensor] = []
    for token_activations in object_activations:
        all_activations.append(token_activations[0])
    return torch.stack(all_activations).mean(dim=0)


ESTIMATOR_LOOKUP: dict[EstimationMethod, RepresentationEstimator] = {
    "mean": mean_estimator,
    "weighted_mean": weighted_mean_estimator,
    "first_token": first_token_estimator,
}


class MultitokenEstimator:
    model: nn.Module
    tokenizer: Tokenizer
    device: torch.device
    dataset: RelationDataset
    prompt_generator: PromptGenerator
    layer_matcher: LayerMatcher
    prompt_validator: PromptValidator

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        layer_matcher: LayerMatcher,
        dataset: RelationDataset,
        prompt_validator: Optional[PromptValidator] = None,
        device: torch.device = DEFAULT_DEVICE,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dataset = dataset
        self.prompt_generator = PromptGenerator(dataset)
        self.layer_matcher = layer_matcher
        self.prompt_validator = prompt_validator or PromptValidator(model, tokenizer)

    def collect_object_activations(
        self,
        relation_name: str,
        object_name: str,
        num_fsl_examples: int = 5,
        entity_modifiers: list[EntityModifier] = AUGMENTATION_MODIFIERS,
        exclude_fsl_examples_of_object: bool = True,
        batch_size: int = 8,
        max_prompts: Optional[int] = None,
        verbose: bool = True,
        valid_relations: Optional[set[str]] = None,
    ) -> ObjectActivations:
        self.model.eval()
        prompts = list(
            self.prompt_generator.generate_prompts_for_object(
                relation_name=relation_name,
                object_name=object_name,
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
        relation_name: str,
        object_name: str,
        num_fsl_examples: int = 5,
        entity_modifiers: list[EntityModifier] = AUGMENTATION_MODIFIERS,
        exclude_fsl_examples_of_object: bool = True,
        batch_size: int = 8,
        verbose: bool = True,
        estimation_method: EstimationMethod = "mean",
        max_prompts: Optional[int] = None,
        valid_relations: Optional[set[str]] = None,
    ) -> ObjectRepresentation:
        object_activations = self.collect_object_activations(
            relation_name=relation_name,
            object_name=object_name,
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
        entity_modifiers: list[EntityModifier] = AUGMENTATION_MODIFIERS,
        exclude_fsl_examples_of_object: bool = True,
        batch_size: int = 8,
        verbose: bool = True,
        estimation_method: EstimationMethod = "mean",
        max_prompts_per_object: Optional[int] = None,
        min_prompts_per_object: int = 0,
    ) -> list[ObjectRepresentation]:
        samples = self.dataset.get_relation_samples(relation)
        objects = {sample.object for sample in samples}
        object_representations = []
        for i, object in enumerate(objects):
            log_or_print(
                f"estimating representation for {object} ({i+1}/{len(objects)})",
                verbose=verbose,
            )
            if min_prompts_per_object > 0:
                num_prompts = len(
                    self.prompt_generator.generate_prompts_for_object(
                        relation_name=relation,
                        object_name=object,
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
                relation_name=relation,
                object_name=object,
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
def _apply_estimator(
    object_activations: ObjectActivations, estimator: RepresentationEstimator
) -> ObjectRepresentation:
    activations_by_layer: dict[int, list[tuple[torch.Tensor, ...]]] = defaultdict(list)
    for object_activation in object_activations.activations:
        for layer_num, activations in object_activation.layer_activations.items():
            activations_by_layer[layer_num].append(activations)
    return ObjectRepresentation(
        object_name=object_activations.object_name,
        activations=object_activations.activations,
        layer_representations={
            layer_num: estimator(activations)
            for layer_num, activations in activations_by_layer.items()
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
    if isinstance(estimation_method, str):
        estimator = ESTIMATOR_LOOKUP[estimation_method]
    else:
        estimator = estimation_method
    return _apply_estimator(object_activations, estimator)
