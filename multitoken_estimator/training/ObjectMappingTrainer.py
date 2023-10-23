from collections import defaultdict
from typing import Iterable, Optional, Sequence

import pytorch_lightning as pl
import torch
from lightning_fabric.plugins.precision.precision import _PRECISION_INPUT
from pytorch_lightning.loggers.logger import Logger
from tokenizers import Tokenizer
from torch import nn
from torch.utils.data import DataLoader

from multitoken_estimator.data.data_model import RelationDataModel
from multitoken_estimator.data.database import Database
from multitoken_estimator.lib.extract_token_activations import (
    extract_final_token_activations,
    extract_token_activations,
)
from multitoken_estimator.lib.layer_matching import LayerMatcher, get_layer_name
from multitoken_estimator.lib.logger import log_or_print
from multitoken_estimator.lib.PromptValidator import PromptValidator
from multitoken_estimator.lib.token_utils import find_prompt_answer_data
from multitoken_estimator.lib.torch_utils import get_device
from multitoken_estimator.ObjectMappingModel import ObjectMappingModel, prefix_object
from multitoken_estimator.PromptGenerator import (
    DEFAULT_ENTITY_MODIFIERS,
    Prompt,
    PromptGenerator,
)
from multitoken_estimator.training.ObjectMappingDataset import (
    ObjectMappingDataset,
    ObjectMappingSample,
)
from multitoken_estimator.training.ObjectMappingSampleReweighter import (
    ObjectMappingSampleReweighter,
)
from multitoken_estimator.training.ObjectMappingTrainingWrapper import (
    ObjectMappingTrainingWrapper,
)
from multitoken_estimator.training.train_lre import ObjectAggregation


class ObjectMappingTrainer:
    model: nn.Module
    tokenizer: Tokenizer
    layer_matcher: LayerMatcher
    dataset: Database
    prompt_generator: PromptGenerator
    prompt_validator: PromptValidator

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        layer_matcher: LayerMatcher,
        dataset: Database,
        prompt_validator: Optional[PromptValidator] = None,
    ):
        self.dataset = dataset
        self.prompt_generator = PromptGenerator(dataset)
        self.model = model
        self.tokenizer = tokenizer
        self.layer_matcher = layer_matcher
        self.prompt_validator = prompt_validator or PromptValidator(model, tokenizer)

    def train(
        self,
        source_layer: int,
        target_layer: int,
        object_aggregation: ObjectAggregation = "mean",
        n_fsl_prompts: int = 5,
        batch_size: int = 8,
        lr: float = 1e-4,
        lr_gamma: float = 0.9,
        n_epochs: int = 100,
        squeeze_dim: int = 100,
        # TODO: figure this out automatically somehow
        activations_dim: int = 4096,  # This must match the model hidden activations size
        augment_prompts: bool = False,
        verbose: bool = True,
        reweight_samples: bool = True,
        pl_precision: _PRECISION_INPUT = "16-mixed",
        pl_logger: Optional[Logger | Iterable[Logger] | bool] = True,
        use_gpu: bool = torch.cuda.is_available(),
        move_to_cpu: bool = True,
        use_relation_prefixes: bool = True,
    ) -> ObjectMappingModel:
        relation_prefixes: dict[str, str] = {}
        if use_relation_prefixes:
            relations = self.dataset.query_all(RelationDataModel)
            for relation in relations:
                relation_prefixes[relation.name] = _extract_prefix(relation.templates)
        mapping_dataset = self._build_training_dataset(
            source_layer=source_layer,
            target_layer=target_layer,
            object_aggregation=object_aggregation,
            n_fsl_prompts=n_fsl_prompts,
            batch_size=batch_size,
            augment_prompts=augment_prompts,
            verbose=verbose,
            relation_prefixes=relation_prefixes,
        )
        reweighter = ObjectMappingSampleReweighter(mapping_dataset.samples)
        data_loader = DataLoader(mapping_dataset, batch_size=batch_size, shuffle=True)
        object_mapping_model = ObjectMappingModel(
            activations_dim=activations_dim,
            squeeze_dim=squeeze_dim,
            source_layer=source_layer,
            target_layer=target_layer,
            object_aggregation=object_aggregation,
        )
        training_wrapper = ObjectMappingTrainingWrapper(
            object_mapping=object_mapping_model,
            lr=lr,
            lr_gamma=lr_gamma,
            reweighter=reweighter if reweight_samples else None,
        )
        pl_trainer = pl.Trainer(
            max_epochs=n_epochs,
            accelerator="gpu" if use_gpu else "cpu",
            log_every_n_steps=1,
            logger=pl_logger,
            enable_checkpointing=False,
            precision=pl_precision,
        )
        pl_trainer.fit(training_wrapper, data_loader)
        if move_to_cpu:
            object_mapping_model = object_mapping_model.cpu()
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
        relation_prefixes: Optional[dict[str, str]] = None,
    ) -> ObjectMappingDataset:
        entity_modifiers = DEFAULT_ENTITY_MODIFIERS if augment_prompts else None
        raw_prompts = self.prompt_generator.generate_prompts_for_all_relations(
            num_fsl_examples=n_fsl_prompts, entity_modifiers=entity_modifiers
        )
        log_or_print(
            f"Generated {len(raw_prompts)} prompts for building object mapping.",
            verbose=verbose,
        )
        prompts = self.prompt_validator.filter_prompts(
            raw_prompts, batch_size=batch_size, show_progress=verbose
        )
        log_or_print(
            f"Filtered to {len(prompts)} prompts which model can answer correctly.",
            verbose=verbose,
        )

        source_object_activations = self._extract_source_object_final_token_activations(
            prompts=prompts,
            relation_prefixes=relation_prefixes,
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
                            source_vector=source_object_activations[relation_name][
                                object_name
                            ],
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
        prompts: list[Prompt],
        relation_prefixes: dict[str, str] | None,
        layer: int,
        batch_size: int,
        show_progress: bool = False,
        move_to_cpu: bool = True,
    ) -> dict[str, dict[str, torch.Tensor]]:
        layer_name = get_layer_name(self.model, self.layer_matcher, layer)
        results: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        relation_objects = list(
            {(prompt.relation_name, prompt.object_name) for prompt in prompts}
        )
        raw_activations = extract_final_token_activations(
            self.model,
            self.tokenizer,
            layers=[layer_name],
            texts=[
                prefix_object(object_name, relation_name, relation_prefixes)
                for relation_name, object_name in relation_objects
            ],
            device=get_device(self.model),
            batch_size=batch_size,
            show_progress=show_progress,
            move_results_to_cpu=move_to_cpu,
        )
        for (relation, object), layer_activations in zip(
            relation_objects, raw_activations
        ):
            results[relation][object] = layer_activations[layer_name]
        return results


def _extract_prefix(templates: Iterable[str]) -> str:
    # TODO: use all templates rather than just picking one
    return list(templates)[0].replace("{}", "").replace("  ", " ")
