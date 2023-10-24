import os
from typing import Optional

import torch
from tokenizers import Tokenizer
from torch import nn

from multitoken_estimator.data.database import Database
from multitoken_estimator.lib.layer_matching import LayerMatcher
from multitoken_estimator.lib.logger import log_or_print, logger
from multitoken_estimator.lib.PromptValidator import PromptValidator
from multitoken_estimator.RelationalConceptEstimator import RelationalConceptEstimator
from multitoken_estimator.training.LreTrainer import LreTrainer
from multitoken_estimator.training.ObjectMappingTrainer import ObjectMappingTrainer
from multitoken_estimator.training.train_lre import ObjectAggregation


class RelationalConceptEstimatorTrainer:
    """
    Wrapper class around LreTrainer + ObjectMapping Trainer
    """

    database: Database
    lre_trainer: LreTrainer
    object_mapping_trainer: ObjectMappingTrainer

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        layer_matcher: LayerMatcher,
        database: Database,
        prompt_validator: Optional[PromptValidator] = None,
    ):
        self.database = database
        self.lre_trainer = LreTrainer(
            model, tokenizer, layer_matcher, database, prompt_validator
        )
        self.object_mapping_trainer = ObjectMappingTrainer(
            model, tokenizer, layer_matcher, database, prompt_validator
        )

    def train_all(
        self,
        subject_layer: int,
        object_layer: int,
        mapping_source_layer: int,
        n_lre_training_prompts: int = 5,
        augment_lre_prompts: bool = False,
        object_aggregation: ObjectAggregation = "mean",
        n_fsl_prompts: int = 5,
        batch_size: int = 8,
        max_consider_prompts: int = 100,
        inv_lre_rank: int = 100,
        object_mapping_lr: float = 0.01,
        object_mapping_epochs: int = 100,
        object_mapping_reweight_samples: bool = True,
        activations_dim: int = 4096,
        verbose: bool = True,
        filter_lre_training_prompts: bool = True,
        save_progress_path: Optional[str] = None,
        force_retrain_all: bool = False,
    ) -> list[RelationalConceptEstimator]:
        relations = self.database.get_relation_names()
        estimators = load_trained_estimators(
            save_progress_path, force_retrain_all, verbose=verbose
        )
        already_trained_relations = {
            estimator.inv_lre.relation for estimator in estimators
        }
        for relation in relations:
            if relation in already_trained_relations:
                log_or_print(f"Skipping {relation}, already trained", verbose=verbose)
                continue
            log_or_print(f"Training LRE for {relation}", verbose=verbose)
            try:
                estimator = self.train(
                    relation=relation,
                    subject_layer=subject_layer,
                    object_layer=object_layer,
                    mapping_source_layer=mapping_source_layer,
                    n_lre_training_prompts=n_lre_training_prompts,
                    inv_lre_rank=inv_lre_rank,
                    object_mapping_lr=object_mapping_lr,
                    object_mapping_epochs=object_mapping_epochs,
                    object_mapping_reweight_samples=object_mapping_reweight_samples,
                    activations_dim=activations_dim,
                    augment_lre_prompts=augment_lre_prompts,
                    object_aggregation=object_aggregation,
                    n_fsl_prompts=n_fsl_prompts,
                    batch_size=batch_size,
                    max_consider_prompts=max_consider_prompts,
                    filter_lre_training_prompts=filter_lre_training_prompts,
                    verbose=verbose,
                )
                estimators.append(estimator)
                if save_progress_path is not None:
                    torch.save(estimators, save_progress_path)
            # catch all excepts except cuda OOM
            except Exception as e:
                # if it's OOM, just quit since it won't get better
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    raise e
                logger.exception(f"Error training estimator for relation {relation}")
        return estimators

    def train(
        self,
        relation: str,
        subject_layer: int,
        object_layer: int,
        mapping_source_layer: int,
        n_lre_training_prompts: int = 5,
        augment_lre_prompts: bool = False,
        object_aggregation: ObjectAggregation = "mean",
        n_fsl_prompts: int = 5,
        batch_size: int = 8,
        max_consider_prompts: int = 100,
        inv_lre_rank: int = 100,
        object_mapping_lr: float = 0.01,
        object_mapping_epochs: int = 100,
        object_mapping_reweight_samples: bool = True,
        activations_dim: int = 4096,
        verbose: bool = True,
        filter_lre_training_prompts: bool = True,
    ) -> RelationalConceptEstimator:
        lre = self.lre_trainer.train(
            relation=relation,
            subject_layer=subject_layer,
            object_layer=object_layer,
            n_lre_training_prompts=n_lre_training_prompts,
            augment_lre_prompts=augment_lre_prompts,
            object_aggregation=object_aggregation,
            n_fsl_prompts=n_fsl_prompts,
            batch_size=batch_size,
            max_consider_prompts=max_consider_prompts,
            verbose=verbose,
            filter_training_prompts=filter_lre_training_prompts,
        )
        inv_lre = lre.invert(inv_lre_rank)
        object_mapping = self.object_mapping_trainer.train(
            inv_lre=inv_lre,
            source_layer=mapping_source_layer,
            n_fsl_prompts=n_fsl_prompts,
            batch_size=batch_size,
            lr=object_mapping_lr,
            n_epochs=object_mapping_epochs,
            activations_dim=activations_dim,
            reweight_samples=object_mapping_reweight_samples,
        )
        return RelationalConceptEstimator(inv_lre, object_mapping)


def load_trained_estimators(
    save_progress_path: Optional[str] = None,
    force_retrain_all: bool = False,
    verbose: bool = True,
) -> list[RelationalConceptEstimator]:
    estimators: list[RelationalConceptEstimator] = []
    if (
        save_progress_path is not None
        and os.path.exists(save_progress_path)
        and not force_retrain_all
    ):
        estimators = torch.load(save_progress_path)
        log_or_print(
            f"Loaded {len(estimators)} estimators from {save_progress_path}",
            verbose,
        )
    return estimators
