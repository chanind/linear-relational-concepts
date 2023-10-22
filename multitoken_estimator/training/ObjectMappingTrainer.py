from collections import defaultdict
from typing import NamedTuple, Sequence

import torch
import torch.optim as optim
from tokenizers import Tokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from multitoken_estimator.data.database import Database
from multitoken_estimator.lib.extract_token_activations import (
    extract_final_token_activations,
    extract_token_activations,
)
from multitoken_estimator.lib.layer_matching import LayerMatcher, get_layer_name
from multitoken_estimator.lib.logger import log_or_print
from multitoken_estimator.lib.token_utils import find_prompt_answer_data
from multitoken_estimator.lib.torch_utils import get_device
from multitoken_estimator.lib.verify_answers_match_expected import (
    verify_answers_match_expected,
)
from multitoken_estimator.ObjectMappingModel import ObjectMappingModel
from multitoken_estimator.PromptGenerator import (
    DEFAULT_ENTITY_MODIFIERS,
    Prompt,
    PromptGenerator,
)
from multitoken_estimator.training.train_lre import ObjectAggregation


class ObjectMappingSample(NamedTuple):
    relation_name: str
    object_name: str
    source_vector: torch.Tensor
    target_vector: torch.Tensor


class ObjectMappingDataset(Dataset[ObjectMappingSample]):
    samples: list[ObjectMappingSample]

    def __init__(self, samples: list[ObjectMappingSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ObjectMappingSample:
        return self.samples[idx]


class ObjectMappingTrainer:
    model: nn.Module
    tokenizer: Tokenizer
    layer_matcher: LayerMatcher
    dataset: Database
    prompt_generator: PromptGenerator

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        layer_matcher: LayerMatcher,
        dataset: Database,
    ):
        self.dataset = dataset
        self.prompt_generator = PromptGenerator(dataset)
        self.model = model
        self.tokenizer = tokenizer
        self.layer_matcher = layer_matcher

    def train(
        self,
        source_layer: int,
        target_layer: int,
        object_aggregation: ObjectAggregation = "mean",
        n_fsl_prompts: int = 5,
        batch_size: int = 8,
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        squeeze_dim: int = 100,
        # TODO: figure this out automatically somehow
        activations_dim: int = 4096,  # This must match the model hidden activations size
        augment_prompts: bool = False,
        verbose: bool = True,
        reweight_samples: bool = True,
    ) -> ObjectMappingModel:
        mapping_dataset = self._build_training_dataset(
            source_layer=source_layer,
            target_layer=target_layer,
            object_aggregation=object_aggregation,
            n_fsl_prompts=n_fsl_prompts,
            batch_size=batch_size,
            augment_prompts=augment_prompts,
            verbose=verbose,
        )
        reweighter = SampleReweighter(mapping_dataset.samples)
        data_loader = DataLoader(mapping_dataset, batch_size=batch_size, shuffle=True)
        object_mapping_model = ObjectMappingModel(
            activations_dim=activations_dim,
            squeeze_dim=squeeze_dim,
            source_layer=source_layer,
            target_layer=target_layer,
            object_aggregation=object_aggregation,
        )
        criterion = nn.MSELoss(reduction="none")
        optimizer = optim.Adam(object_mapping_model.parameters(), lr=learning_rate)

        # Modified training loop
        for epoch in range(n_epochs):
            for (
                relation_names,
                object_names,
                source_vectors,
                target_vectors,
            ) in data_loader:
                # Forward pass
                predictions = object_mapping_model(source_vectors)
                loss = criterion(predictions, target_vectors)
                if reweight_samples:
                    reweighting_tensor = reweighter.calc_samples_reweighting_tensor(
                        relation_names, object_names, loss.device, loss.dtype
                    )
                    loss = loss * reweighting_tensor.unsqueeze(dim=1)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                log_or_print(
                    f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}",
                    verbose=verbose,
                )
        return object_mapping_model

    def _build_training_dataset(
        self,
        source_layer: int,
        target_layer: int,
        object_aggregation: ObjectAggregation = "mean",
        n_fsl_prompts: int = 5,
        batch_size: int = 8,
        augment_prompts: bool = False,
        verbose: bool = True,
    ) -> ObjectMappingDataset:
        entity_modifiers = DEFAULT_ENTITY_MODIFIERS if augment_prompts else None
        raw_prompts = self.prompt_generator.generate_prompts_for_all_relations(
            num_fsl_examples=n_fsl_prompts, entity_modifiers=entity_modifiers
        )
        log_or_print(
            f"Generated {len(raw_prompts)} prompts for building object mapping.",
            verbose=verbose,
        )
        check_prompt_results = verify_answers_match_expected(
            self.model,
            self.tokenizer,
            prompts=[prompt.text for prompt in raw_prompts],
            expected_answers=[prompt.answer for prompt in raw_prompts],
            batch_size=batch_size,
        )
        prompts = [
            prompt
            for prompt, result in zip(raw_prompts, check_prompt_results)
            if result.answer_matches_expected
        ]
        log_or_print(
            f"Filtered to {len(prompts)} prompts which model can answer correctly.",
            verbose=verbose,
        )

        source_object_activations = self._extract_source_object_final_token_activations(
            source_objects=list({prompt.object_name for prompt in prompts}),
            layer=source_layer,
            batch_size=batch_size,
            show_progress=verbose,
        )

        target_activations_by_relation_and_object = (
            self._extract_target_object_activations_by_relation_and_object(
                prompts=prompts,
                layer=target_layer,
                aggregation=object_aggregation,
                batch_size=batch_size,
                show_progress=verbose,
            )
        )

        samples: list[ObjectMappingSample] = []
        for (
            relation_name,
            activations_by_object,
        ) in target_activations_by_relation_and_object.items():
            for object_name, activations in activations_by_object.items():
                for activation in activations:
                    samples.append(
                        ObjectMappingSample(
                            relation_name=relation_name,
                            object_name=object_name,
                            source_vector=source_object_activations[object_name],
                            target_vector=activation,
                        )
                    )
        return ObjectMappingDataset(samples)

    @torch.no_grad()
    def _extract_target_object_activations_by_relation_and_object(
        self,
        prompts: Sequence[Prompt],
        layer: int,
        aggregation: ObjectAggregation,
        batch_size: int,
        show_progress: bool = False,
        move_to_cpu: bool = True,
    ) -> dict[str, dict[str, list[torch.Tensor]]]:
        activations_by_relation_and_object: dict[
            str, dict[str, list[torch.Tensor]]
        ] = defaultdict(lambda: defaultdict(list))
        prompt_answer_data = [
            find_prompt_answer_data(self.tokenizer, prompt.text, prompt.answer)
            for prompt in prompts
        ]
        layer_name = get_layer_name(self.model, self.layer_matcher, layer)
        raw_activations = extract_token_activations(
            self.model,
            self.tokenizer,
            layers=[layer_name],
            texts=[prompt_answer.full_prompt for prompt_answer in prompt_answer_data],
            token_indices=[
                prompt_answer.output_answer_token_indices
                for prompt_answer in prompt_answer_data
            ],
            device=get_device(self.model),
            batch_size=batch_size,
            show_progress=show_progress,
            move_results_to_cpu=move_to_cpu,
        )
        for prompt, raw_activation in zip(prompts, raw_activations):
            if aggregation == "mean":
                activation = torch.stack(raw_activation[layer_name]).mean(dim=0)
            elif aggregation == "first_token":
                activation = raw_activation[layer_name][0]
            else:
                raise ValueError(f"Unknown object aggregation: {aggregation}")
            activations_by_relation_and_object[prompt.relation_name][
                prompt.object_name
            ].append(activation)
        return activations_by_relation_and_object

    @torch.no_grad()
    def _extract_source_object_final_token_activations(
        self,
        source_objects: Sequence[str],
        layer: int,
        batch_size: int,
        show_progress: bool = False,
        move_to_cpu: bool = True,
    ) -> dict[str, torch.Tensor]:
        layer_name = get_layer_name(self.model, self.layer_matcher, layer)
        raw_activations = extract_final_token_activations(
            self.model,
            self.tokenizer,
            layers=[layer_name],
            texts=source_objects,
            device=get_device(self.model),
            batch_size=batch_size,
            show_progress=show_progress,
            move_results_to_cpu=move_to_cpu,
        )
        return {
            object: raw_activations[layer_name]
            for object, raw_activations in zip(source_objects, raw_activations)
        }


class SampleReweighter:
    _counts_by_relation_and_object: dict[str, dict[str, int]]
    _num_objects_per_relation: dict[str, int]
    _reweighting_cache: dict[tuple[str, str], float]

    def __init__(self, samples: Sequence[ObjectMappingSample]):
        self._reweighting_cache = {}
        self._counts_by_relation_and_object = defaultdict(lambda: defaultdict(int))
        for sample in samples:
            self._counts_by_relation_and_object[sample.relation_name][
                sample.object_name
            ] += 1
        self._num_objects_per_relation = {
            relation_name: len(objects)
            for relation_name, objects in self._counts_by_relation_and_object.items()
        }

    def calc_samples_reweighting_tensor(
        self,
        relation_names: Sequence[str],
        object_names: Sequence[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        assert len(relation_names) == len(object_names)

        return torch.tensor(
            [
                self.calc_sample_reweighting(relation_name, object_name)
                for relation_name, object_name in zip(relation_names, object_names)
            ],
            device=device,
            dtype=dtype,
        )

    def calc_sample_reweighting(self, relation_name: str, object_name: str) -> float:
        """
        Reweight first for the object frequency in the relation, then for the
        number of objects per relation
        """
        cache_key = (relation_name, object_name)
        if cache_key not in self._reweighting_cache:
            num_samples_in_relation = sum(
                self._counts_by_relation_and_object[relation_name].values()
            )
            object_portion = (
                self._counts_by_relation_and_object[relation_name][object_name]
                / num_samples_in_relation
            )
            object_reweighting = _calc_reweighting(
                object_portion, self._num_objects_per_relation[relation_name]
            )

            num_objects_total = sum(self._num_objects_per_relation.values())
            num_relations = len(self._num_objects_per_relation)
            relation_portion = (
                self._num_objects_per_relation[relation_name] / num_objects_total
            )
            relation_reweighting = _calc_reweighting(relation_portion, num_relations)
            reweighting = object_reweighting * relation_reweighting
            self._reweighting_cache[cache_key] = reweighting
        return self._reweighting_cache[cache_key]


def _calc_reweighting(class_portion: float, num_classes: int) -> float:
    """
    Calculate reweighting to increase weight of underrepresented classes.
    Use 1 / (num_classes * class_portion) as the weighting. This is based
    on the "balanced" weighting from scikit-learn's LinearSVC class
    """
    if class_portion == 0:
        return 0
    return 1 / (num_classes * class_portion)
