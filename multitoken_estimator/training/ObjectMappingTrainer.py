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
from multitoken_estimator.LinearRelationalEmbedding import (
    InvertedLinearRelationalEmbedding,
)
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
        inv_lre: InvertedLinearRelationalEmbedding,
        source_layer: int,
        n_fsl_prompts: int = 5,
        batch_size: int = 8,
        lr: float = 1e-4,
        lr_gamma: float = 0.9,
        n_epochs: int = 100,
        # TODO: figure this out automatically somehow
        activations_dim: int = 4096,  # This must match the model hidden activations size
        augment_prompts: bool = False,
        verbose: bool = True,
        reweight_samples: bool = True,
        pl_precision: _PRECISION_INPUT = "16-mixed",
        pl_logger: Optional[Logger | Iterable[Logger] | bool] = True,
        use_gpu: bool = torch.cuda.is_available(),
        move_to_cpu: bool = True,
        use_relation_prefix: bool = True,
        validation_dataset: Optional[Database] = None,
    ) -> ObjectMappingModel:
        prefix: str | None = None
        if use_relation_prefix:
            relation = self.dataset.query_one_or_throw(
                RelationDataModel, lambda rel: rel.name == inv_lre.relation
            )
            prefix = _extract_prefix(relation.templates)
        mapping_dataset = self._build_training_dataset(
            inv_lre=inv_lre,
            prompt_generator=self.prompt_generator,
            source_layer=source_layer,
            n_fsl_prompts=n_fsl_prompts,
            batch_size=batch_size,
            augment_prompts=augment_prompts,
            verbose=verbose,
            prefix=prefix,
        )
        val_loader = None
        if validation_dataset is not None:
            val_prompt_generator = PromptGenerator(validation_dataset)
            val_mapping_dataset = self._build_training_dataset(
                inv_lre=inv_lre,
                prompt_generator=val_prompt_generator,
                source_layer=source_layer,
                n_fsl_prompts=n_fsl_prompts,
                batch_size=batch_size,
                augment_prompts=augment_prompts,
                verbose=verbose,
                prefix=prefix,
            )
            val_loader = DataLoader(val_mapping_dataset, batch_size=batch_size)
        reweighter = ObjectMappingSampleReweighter(mapping_dataset.samples)
        data_loader = DataLoader(mapping_dataset, batch_size=batch_size, shuffle=True)
        object_mapping_model = ObjectMappingModel(
            activations_dim=activations_dim,
            target_rank=inv_lre.rank,
            source_layer=source_layer,
            target_layer=inv_lre.object_layer,
            object_aggregation=inv_lre.object_aggregation,
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
        pl_trainer.fit(training_wrapper, data_loader, val_dataloaders=val_loader)
        if move_to_cpu:
            object_mapping_model = object_mapping_model.cpu()
        return object_mapping_model

    def _build_training_dataset(
        self,
        inv_lre: InvertedLinearRelationalEmbedding,
        prompt_generator: PromptGenerator,
        source_layer: int,
        n_fsl_prompts: int = 5,
        batch_size: int = 8,
        augment_prompts: bool = False,
        verbose: bool = True,
        prefix: Optional[str] = None,
    ) -> ObjectMappingDataset:
        entity_modifiers = DEFAULT_ENTITY_MODIFIERS if augment_prompts else None
        raw_prompts = prompt_generator.generate_prompts_for_relation(
            inv_lre.relation,
            num_fsl_examples=n_fsl_prompts,
            entity_modifiers=entity_modifiers,
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
            prefix=prefix,
            layer=source_layer,
            batch_size=batch_size,
            show_progress=verbose,
        )

        target_activations_by_object = (
            self._extract_target_object_activations_for_inv_lre(
                inv_lre=inv_lre,
                prompts=prompts,
                batch_size=batch_size,
                show_progress=verbose,
            )
        )

        samples: list[ObjectMappingSample] = []
        for (
            object_name,
            activations,
        ) in target_activations_by_object.items():
            for activation in activations:
                samples.append(
                    ObjectMappingSample(
                        object_name=object_name,
                        source_vector=source_object_activations[object_name],
                        target_vector=activation,
                    )
                )
        return ObjectMappingDataset(samples)

    @torch.no_grad()
    def _extract_target_object_activations_for_inv_lre(
        self,
        inv_lre: InvertedLinearRelationalEmbedding,
        prompts: Sequence[Prompt],
        batch_size: int,
        show_progress: bool = False,
        move_to_cpu: bool = True,
    ) -> dict[str, list[torch.Tensor]]:
        activations_by_object: dict[str, list[torch.Tensor]] = defaultdict(list)
        prompt_answer_data = [
            find_prompt_answer_data(self.tokenizer, prompt.text, prompt.answer)
            for prompt in prompts
        ]
        layer_name = get_layer_name(
            self.model, self.layer_matcher, inv_lre.object_layer
        )
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
            if inv_lre.object_aggregation == "mean":
                activation = torch.stack(raw_activation[layer_name]).mean(dim=0)
            elif inv_lre.object_aggregation == "first_token":
                activation = raw_activation[layer_name][0]
            else:
                raise ValueError(
                    f"Unknown inv_lre.object_aggregation: {inv_lre.object_aggregation}"
                )
            activations_by_object[prompt.object_name].append(
                inv_lre.calc_low_rank_target_activation(activation)
            )
        return activations_by_object

    @torch.no_grad()
    def _extract_source_object_final_token_activations(
        self,
        prompts: list[Prompt],
        prefix: str | None,
        layer: int,
        batch_size: int,
        show_progress: bool = False,
        move_to_cpu: bool = True,
    ) -> dict[str, torch.Tensor]:
        layer_name = get_layer_name(self.model, self.layer_matcher, layer)
        results: dict[str, torch.Tensor] = {}
        object_names = list({prompt.object_name for prompt in prompts})
        raw_activations = extract_final_token_activations(
            self.model,
            self.tokenizer,
            layers=[layer_name],
            texts=[prefix_object(object_name, prefix) for object_name in object_names],
            device=get_device(self.model),
            batch_size=batch_size,
            show_progress=show_progress,
            move_results_to_cpu=move_to_cpu,
        )
        for object_name, layer_activations in zip(object_names, raw_activations):
            results[object_name] = layer_activations[layer_name]
        return results


def _extract_prefix(templates: Iterable[str]) -> str:
    # TODO: use all templates rather than just picking one
    return list(templates)[0].replace("{}", "").replace("  ", " ")
