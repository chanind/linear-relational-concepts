import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Generic, Optional, TypeVar

import torch
from linear_relational import Concept
from tokenizers import Tokenizer
from torch import nn

from linear_relational_concepts.data.RelationDataset import RelationDataset
from linear_relational_concepts.lib.layer_matching import LayerMatcher
from linear_relational_concepts.lib.logger import log_or_print, logger
from linear_relational_concepts.lib.PromptValidator import PromptValidator
from linear_relational_concepts.PromptGenerator import PromptGenerator


@dataclass
class ConceptTrainerOptions:
    layer: int


T = TypeVar("T", bound="ConceptTrainerOptions")


class ConceptTrainer(ABC, Generic[T]):
    model: nn.Module
    tokenizer: Tokenizer
    layer_matcher: LayerMatcher
    dataset: RelationDataset
    prompt_generator: PromptGenerator
    prompt_validator: PromptValidator

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        layer_matcher: LayerMatcher,
        dataset: RelationDataset,
        prompt_validator: Optional[PromptValidator] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_matcher = layer_matcher
        self.dataset = dataset
        self.prompt_generator = PromptGenerator(dataset)
        self.prompt_validator = prompt_validator or PromptValidator(model, tokenizer)

    @property
    def relation_category_map(self) -> dict[str, str]:
        return {
            relation.name: relation.category
            for relation in self.dataset.relations
            if relation.category is not None
        }

    def train_all(
        self,
        opts: T,
        save_progress_path: Optional[str | Path] = None,
        force_retrain_all: bool = False,
        verbose: bool = True,
    ) -> list[Concept]:
        relations = self.dataset.get_relation_names()
        concepts = load_trained_concepts(
            save_progress_path, force_retrain_all, verbose=verbose
        )
        already_trained_relations = {lre.relation for lre in concepts}
        for i, relation in enumerate(relations):
            if relation in already_trained_relations:
                log_or_print(f"Skipping {relation}, already trained", verbose=verbose)
                continue
            log_or_print(
                f"Training relation {relation} ({i+1}/{len(relations)})",
                verbose=verbose,
            )
            try:
                start_time = time()
                relation_concepts = self.train_relation_concepts(
                    relation=relation,
                    opts=opts,
                    verbose=verbose,
                )
                log_or_print(
                    f"Trained {len(relation_concepts)} concepts for {relation} in {time() - start_time:.2f}s",
                    verbose=verbose,
                )
                concepts.extend(relation_concepts)
                if save_progress_path is not None:
                    torch.save(concepts, save_progress_path)
            # catch all excepts except cuda OOM
            except Exception as e:
                # if it's OOM, just quit since it won't get better
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    raise e
                logger.exception(f"Error training relation {relation}")
        return concepts

    @abstractmethod
    def train_relation_concepts(
        self,
        relation: str,
        opts: T,
        verbose: bool = True,
    ) -> list[Concept]:
        pass


def load_trained_concepts(
    save_progress_path: Optional[str | Path] = None,
    force_retrain_all: bool = False,
    verbose: bool = True,
) -> list[Concept]:
    concepts: list[Concept] = []
    if (
        save_progress_path is not None
        and os.path.exists(save_progress_path)
        and not force_retrain_all
    ):
        concepts = torch.load(save_progress_path)
        log_or_print(
            f"Loaded {len(concepts)} concepts from {save_progress_path}",
            verbose,
        )
    return concepts
