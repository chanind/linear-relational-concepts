import os
from typing import Optional

import torch
from tokenizers import Tokenizer
from torch import nn

from multitoken_estimator.database import Database
from multitoken_estimator.lib.layer_matching import LayerMatcher
from multitoken_estimator.lib.logger import log_or_print, logger
from multitoken_estimator.lib.util import sample_or_all
from multitoken_estimator.lib.verify_answers_match_expected import (
    verify_answers_match_expected,
)
from multitoken_estimator.LinearRelationalEmbedding import LinearRelationalEmbedding
from multitoken_estimator.PromptGenerator import (
    DEFAULT_ENTITY_MODIFIERS,
    PromptGenerator,
)
from multitoken_estimator.training.train_lre import ObjectAggregation, train_lre


class LreTrainer:
    model: nn.Module
    tokenizer: Tokenizer
    layer_matcher: LayerMatcher
    database: Database
    prompt_generator: PromptGenerator

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        layer_matcher: LayerMatcher,
        database: Database,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_matcher = layer_matcher
        self.database = database
        self.prompt_generator = PromptGenerator(database)

    def train_all_relations(
        self,
        subject_layer: int,
        object_layer: int,
        n_lre_training_prompts: int = 5,
        augment_lre_prompts: bool = False,
        object_aggregation: ObjectAggregation = "mean",
        n_fsl_prompts: int = 5,
        batch_size: int = 8,
        max_consider_prompts: int = 100,
        verbose: bool = True,
        filter_training_prompts: bool = True,
        save_progress_path: Optional[str] = None,
        force_retrain_all: bool = False,
    ) -> list[LinearRelationalEmbedding]:
        relations = self.database.get_relation_names()
        lres = load_trained_lres(save_progress_path, force_retrain_all, verbose=verbose)
        already_trained_relations = {lre.relation for lre in lres}
        for relation in relations:
            if relation in already_trained_relations:
                log_or_print(f"Skipping {relation}, already trained", verbose=verbose)
                continue
            log_or_print(f"Training LRE for {relation}", verbose=verbose)
            try:
                lre = self.train_relation_lre(
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
                    filter_training_prompts=filter_training_prompts,
                )
                lres.append(lre)
                if save_progress_path is not None:
                    torch.save(lres, save_progress_path)
            except Exception:
                logger.exception(f"Error training relation {relation}")
        return lres

    def train_relation_lre(
        self,
        relation: str,
        subject_layer: int,
        object_layer: int,
        n_lre_training_prompts: int = 5,
        augment_lre_prompts: bool = False,
        object_aggregation: ObjectAggregation = "mean",
        n_fsl_prompts: int = 5,
        batch_size: int = 8,
        max_consider_prompts: int = 100,
        verbose: bool = True,
        filter_training_prompts: bool = True,
    ) -> LinearRelationalEmbedding:
        lre_modifiers = DEFAULT_ENTITY_MODIFIERS if augment_lre_prompts else None
        relation_prompts = list(
            self.prompt_generator.generate_prompts_for_relation(
                relation_name=relation,
                num_fsl_examples=n_fsl_prompts,
                entity_modifiers=lre_modifiers,
            )
        )
        relation_prompts = sample_or_all(relation_prompts, max_consider_prompts)
        if filter_training_prompts:
            check_prompt_results = verify_answers_match_expected(
                self.model,
                self.tokenizer,
                prompts=[prompt.text for prompt in relation_prompts],
                expected_answers=[prompt.answer for prompt in relation_prompts],
                batch_size=batch_size,
            )
            relation_prompts = [
                prompt
                for prompt, result in zip(relation_prompts, check_prompt_results)
                if result.answer_matches_expected
            ]
        if len(relation_prompts) == 0:
            raise ValueError(f"No valid prompts found for {relation}.")
        log_or_print(f"Training LRE for {relation}", verbose=verbose)
        training_prompts = sample_or_all(list(relation_prompts), n_lre_training_prompts)
        return train_lre(
            self.model,
            self.tokenizer,
            layer_matcher=self.layer_matcher,
            relation_name=relation,
            subject_layer=subject_layer,
            object_layer=object_layer,
            prompts=training_prompts,
            object_aggregation=object_aggregation,
            move_to_cpu=False,
        )


def load_trained_lres(
    save_progress_path: Optional[str] = None,
    force_retrain_all: bool = False,
    verbose: bool = True,
) -> list[LinearRelationalEmbedding]:
    lres: list[LinearRelationalEmbedding] = []
    if (
        save_progress_path is not None
        and os.path.exists(save_progress_path)
        and not force_retrain_all
    ):
        lres = torch.load(save_progress_path)
        log_or_print(
            f"Loaded {len(lres)} lres from {save_progress_path}",
            verbose,
        )
    return lres
