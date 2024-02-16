from dataclasses import dataclass, field
from math import ceil, floor
from time import time
from typing import Literal

from linear_relational import Concept, InvertedLre, ObjectAggregation, Prompt
from linear_relational import Trainer as LreTrainer
from linear_relational.lib.balance_grouped_items import balance_grouped_items
from linear_relational.lib.util import group_items, sample_or_all
from typing_extensions import override

from linear_relational_concepts.lib.logger import logger
from linear_relational_concepts.PromptGenerator import AUGMENTATION_MODIFIERS
from linear_relational_concepts.training.ConceptTrainer import (
    ConceptTrainer,
    ConceptTrainerOptions,
)

VectorAggregation = Literal["pre_mean", "post_mean"]
LreSamplingMethod = Literal[
    "random",
    "only_current_object",
    "exclude_current_object",
    "balanced_ceil",
    "balanced_floor",
]


@dataclass
class LreConceptTrainerOptions(ConceptTrainerOptions):
    layer: int
    object_layer: int
    inv_lre_rank: int = 100
    max_lre_training_samples: int = 5
    augment_prompts: bool = False
    object_aggregation: ObjectAggregation = "mean"
    vector_aggregation: VectorAggregation = "post_mean"
    sampling_method: LreSamplingMethod = "random"
    exclude_fsl_examples_of_object: bool = False
    n_fsl_prompts: int = 4
    batch_size: int = 8
    filter_training_prompts: bool = True
    seed: int | str | float = 42


class LreConceptTrainer(ConceptTrainer[LreConceptTrainerOptions]):
    @override
    def train_relation_concepts(
        self,
        relation: str,
        opts: LreConceptTrainerOptions,
        verbose: bool = True,
    ) -> list[Concept]:
        lre_trainer = LreTrainer(
            self.model,
            self.tokenizer,
            layer_matcher=self.layer_matcher,
            prompt_validator=self.prompt_validator,
        )

        lre_modifiers = AUGMENTATION_MODIFIERS if opts.augment_prompts else None
        relation_prompts = list(
            self.prompt_generator.generate_prompts_for_relation(
                relation_name=relation,
                num_fsl_examples=opts.n_fsl_prompts,
                entity_modifiers=lre_modifiers,
                exclude_fsl_examples_of_object=opts.exclude_fsl_examples_of_object,
            )
        )
        if opts.filter_training_prompts:
            relation_prompts = self.prompt_validator.filter_prompts(
                relation_prompts, batch_size=opts.batch_size, show_progress=verbose
            )
        prompts_by_object = group_items(relation_prompts, lambda p: p.object_name)
        if len(relation_prompts) == 0:
            raise ValueError(f"No valid prompts found for {relation}.")
        inv_lre_run_manager = InvLreTrainingRunManager(
            lre_trainer=lre_trainer,
            max_train_samples=opts.max_lre_training_samples,
            sampling_method=opts.sampling_method,
            relation_name=relation,
            subject_layer=opts.layer,
            object_layer=opts.object_layer,
            object_aggregation=opts.object_aggregation,
            prompts_by_object=prompts_by_object,
            inv_lre_rank=opts.inv_lre_rank,
            seed=opts.seed,
        )
        return lre_trainer.train_relation_concepts_from_inv_lre(
            relation=relation,
            inv_lre=lambda object_name: inv_lre_run_manager.get_inv_lre_for_object(
                object_name
            ),
            prompts=relation_prompts,
            object_layer=opts.object_layer,
            object_aggregation=opts.object_aggregation,
            vector_aggregation=opts.vector_aggregation,
            validate_prompts_batch_size=opts.batch_size,
            extract_objects_batch_size=opts.batch_size,
            validate_prompts=False,  # this is alreay done above
            verbose=verbose,
        )


@dataclass
class InvLreTrainingRunManager:
    """
    A helper class to train inverted LREs for a relation, keeping track of which LREs can be cached and re-used.
    A LRE for concept A can be reused for concept B if the prompts LRE A was trained on match the
    n_lre_non_concept_train_samples and n_lre_concept_train_samples requirements for LRE B as well.
    This class exists because training Inv LREs is expensive so we should avoid it unnecessarily
    """

    lre_trainer: LreTrainer
    relation_name: str
    subject_layer: int
    object_layer: int
    object_aggregation: Literal["mean", "first_token"]
    sampling_method: LreSamplingMethod
    max_train_samples: int
    prompts_by_object: dict[str, list[Prompt]]
    inv_lre_rank: int
    seed: int | str | float

    # internally managed state
    _remaining_objects: set[str] = field(init=False)
    _precomputed_inv_lres_by_object: dict[str, InvertedLre] = field(init=False)

    def __post_init__(self) -> None:
        self._remaining_objects = set(self.prompts_by_object.keys())
        self._precomputed_inv_lres_by_object = {}

    def get_inv_lre_for_object(self, object_name: str) -> InvertedLre:
        if object_name not in self._remaining_objects:
            raise ValueError(f"Object {object_name} has already been trained")
        self._remaining_objects.remove(object_name)
        if object_name in self._precomputed_inv_lres_by_object:
            logger.info(f"Using cached LRE for {object_name}")
            inv_lre = self._precomputed_inv_lres_by_object[object_name]
            del self._precomputed_inv_lres_by_object[object_name]
            return inv_lre
        return self._train_inv_lre_for_object(object_name)

    def _train_inv_lre_for_object(self, object_name: str) -> InvertedLre:
        start_time = time()
        train_samples = self._get_train_samples_for_object(object_name)
        inv_lre = self.lre_trainer.train_lre(
            prompts=train_samples,
            relation=self.relation_name,
            subject_layer=self.subject_layer,
            object_layer=self.object_layer,
            object_aggregation=self.object_aggregation,
            validate_prompts=False,
            move_to_cpu=False,
        ).invert(self.inv_lre_rank)
        logger.info(f"Trained LRE for {object_name} in {time() - start_time:.2f}s")
        for remaining_object in self._remaining_objects:
            if self._samples_satisfy_object_constraints(
                remaining_object, train_samples
            ):
                self._precomputed_inv_lres_by_object[remaining_object] = inv_lre
        return inv_lre

    def _get_train_samples_for_object(self, object_name: str) -> list[Prompt]:
        # We can use a proper interface for these sampling methods to avoid if-statements like this,
        # but I'm not sure if it's worth it. Are we going to actually add more ever?
        if self.sampling_method == "random":
            return _sample_random(
                self.prompts_by_object, self.max_train_samples, self.seed
            )
        elif self.sampling_method == "only_current_object":
            return _sample_only_current_object(
                object_name, self.prompts_by_object, self.max_train_samples, self.seed
            )
        elif self.sampling_method == "exclude_current_object":
            return _sample_exclude_current_object(
                object_name, self.prompts_by_object, self.max_train_samples, self.seed
            )
        elif self.sampling_method == "balanced_ceil":
            return _sample_balanced(
                object_name,
                self.prompts_by_object,
                self.max_train_samples,
                seed=self.seed,
                floor_num_obj_samples=False,
            )
        elif self.sampling_method == "balanced_floor":
            return _sample_balanced(
                object_name,
                self.prompts_by_object,
                self.max_train_samples,
                seed=self.seed,
                floor_num_obj_samples=True,
            )
        else:
            raise ValueError(f"Unknown sampling method {self.sampling_method}")

    def _samples_satisfy_object_constraints(
        self,
        object_name: str,
        samples: list[Prompt],
    ) -> bool:
        """
        Check if the list of samples satisfy the constraints of the sampling strategy
        we're using for the given object
        """
        n_object_samples = 0
        n_non_object_samples = 0
        for sample in samples:
            sample_object = sample.object_name
            if sample_object == object_name:
                n_object_samples += 1
            else:
                n_non_object_samples += 1
        if self.sampling_method == "random":
            return True
        elif self.sampling_method == "only_current_object":
            return n_non_object_samples == 0
        elif self.sampling_method == "exclude_current_object":
            return n_object_samples == 0
        else:
            use_floor = self.sampling_method == "balanced_floor"
            num_obj_samples, num_non_obj_samples = _num_balanced_sample_targets(
                self.prompts_by_object,
                self.max_train_samples,
                floor_num_obj_samples=use_floor,
            )
            return (
                n_object_samples == num_obj_samples
                and n_non_object_samples == num_non_obj_samples
            )


def _sample_random(
    prompts_by_object: dict[str, list[Prompt]],
    max_train_samples: int,
    seed: int | float | str,
) -> list[Prompt]:
    return balance_grouped_items(
        items_by_group=prompts_by_object,
        max_total=max_train_samples,
        seed=seed,
    )


def _sample_only_current_object(
    current_object: str,
    prompts_by_object: dict[str, list[Prompt]],
    max_train_samples: int,
    seed: int | float | str,
) -> list[Prompt]:
    return sample_or_all(
        prompts_by_object[current_object],
        max_train_samples,
        seed=seed,
    )


def _sample_exclude_current_object(
    current_object: str,
    prompts_by_object: dict[str, list[Prompt]],
    max_train_samples: int,
    seed: int | float | str,
) -> list[Prompt]:
    return balance_grouped_items(
        items_by_group={
            object: prompts
            for object, prompts in prompts_by_object.items()
            if object != current_object
        },
        max_total=max_train_samples,
        seed=seed,
    )


def _num_balanced_sample_targets(
    prompts_by_object: dict[str, list[Prompt]],
    max_train_samples: int,
    floor_num_obj_samples: bool,
) -> tuple[int, int]:
    target_ratio = 1 / len(prompts_by_object)
    if floor_num_obj_samples:
        num_obj_samples = floor(target_ratio * max_train_samples)
    else:
        num_obj_samples = ceil(target_ratio * max_train_samples)
    num_non_obj_samples = max_train_samples - num_obj_samples
    return num_obj_samples, num_non_obj_samples


def _sample_balanced(
    current_object: str,
    prompts_by_object: dict[str, list[Prompt]],
    max_train_samples: int,
    floor_num_obj_samples: bool,
    seed: int | float | str,
) -> list[Prompt]:
    non_obj_samples = _sample_exclude_current_object(
        current_object, prompts_by_object, max_train_samples, seed
    )
    obj_samples = _sample_only_current_object(
        current_object, prompts_by_object, max_train_samples, seed
    )
    num_obj_samples, num_non_obj_samples = _num_balanced_sample_targets(
        prompts_by_object, max_train_samples, floor_num_obj_samples
    )

    return obj_samples[:num_obj_samples] + non_obj_samples[:num_non_obj_samples]
