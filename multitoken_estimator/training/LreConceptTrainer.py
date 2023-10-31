from collections import defaultdict
from dataclasses import asdict, dataclass, field
from math import ceil, floor
from time import time
from typing import Literal

import torch
from tokenizers import Tokenizer
from torch import nn
from typing_extensions import override

from multitoken_estimator.Concept import Concept
from multitoken_estimator.lib.balance_grouped_items import balance_grouped_items
from multitoken_estimator.lib.extract_token_activations import extract_token_activations
from multitoken_estimator.lib.layer_matching import LayerMatcher, get_layer_name
from multitoken_estimator.lib.logger import logger
from multitoken_estimator.lib.token_utils import (
    PromptAnswerData,
    find_prompt_answer_data,
)
from multitoken_estimator.lib.torch_utils import get_device
from multitoken_estimator.lib.util import group_items, sample_or_all
from multitoken_estimator.LinearRelationalEmbedding import (
    InvertedLinearRelationalEmbedding,
)
from multitoken_estimator.PromptGenerator import AUGMENTATION_MODIFIERS, Prompt
from multitoken_estimator.training.ConceptTrainer import (
    ConceptTrainer,
    ConceptTrainerOptions,
)
from multitoken_estimator.training.train_lre import ObjectAggregation, train_lre

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
            model=self.model,
            tokenizer=self.tokenizer,
            max_train_samples=opts.max_lre_training_samples,
            sampling_method=opts.sampling_method,
            hidden_layers_matcher=self.layer_matcher,
            relation_name=relation,
            subject_layer=opts.layer,
            object_layer=opts.object_layer,
            object_aggregation=opts.object_aggregation,
            prompts_by_object=prompts_by_object,
            inv_lre_rank=opts.inv_lre_rank,
            seed=opts.seed,
        )
        start_time = time()
        object_activations = self._extract_target_object_activations_for_inv_lre(
            prompts=relation_prompts,
            batch_size=opts.batch_size,
            object_aggregation=opts.object_aggregation,
            object_layer=opts.object_layer,
            show_progress=verbose,
            move_to_cpu=True,
        )
        logger.info(
            f"Extracted {len(object_activations)} object activations in {time() - start_time:.2f}s"
        )
        concepts: list[Concept] = []

        with torch.no_grad():
            for (
                object_name,
                activations,
            ) in object_activations.items():
                concept = self._build_concept(
                    relation_name=relation,
                    object_name=object_name,
                    activations=activations,
                    inv_lre_run_manager=inv_lre_run_manager,
                    opts=opts,
                )
                concepts.append(concept)
        return concepts

    def _build_concept(
        self,
        relation_name: str,
        object_name: str,
        activations: list[torch.Tensor],
        inv_lre_run_manager: "InvLreTrainingRunManager",
        opts: LreConceptTrainerOptions,
    ) -> Concept:
        inv_lre = inv_lre_run_manager.get_inv_lre_for_object(object_name)
        device = inv_lre.bias.device
        dtype = inv_lre.bias.dtype
        if opts.vector_aggregation == "pre_mean":
            acts = [torch.stack(activations).to(device=device, dtype=dtype).mean(dim=0)]
        elif opts.vector_aggregation == "post_mean":
            acts = [act.to(device=device, dtype=dtype) for act in activations]
        else:
            raise ValueError(
                f"Unknown vector aggregation method {opts.vector_aggregation}"
            )
        vecs = [
            inv_lre.calculate_subject_activation(act, normalize=False) for act in acts
        ]
        vec = torch.stack(vecs).mean(dim=0)
        vec = vec / vec.norm()
        return Concept(
            object=object_name,
            relation=relation_name,
            layer=opts.layer,
            vector=vec.detach().clone().cpu(),
            metadata=asdict(opts),
        )

    @torch.no_grad()
    def _extract_target_object_activations_for_inv_lre(
        self,
        object_layer: int,
        object_aggregation: Literal["mean", "first_token"],
        prompts: list[Prompt],
        batch_size: int,
        show_progress: bool = False,
        move_to_cpu: bool = True,
    ) -> dict[str, list[torch.Tensor]]:
        activations_by_object: dict[str, list[torch.Tensor]] = defaultdict(list)
        prompt_answer_data: list[PromptAnswerData] = []
        for prompt in prompts:
            prompt_answer_data.append(
                find_prompt_answer_data(self.tokenizer, prompt.text, prompt.answer)
            )

        layer_name = get_layer_name(self.model, self.layer_matcher, object_layer)
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
            if object_aggregation == "mean":
                activation = torch.stack(raw_activation[layer_name]).mean(dim=0)
            elif object_aggregation == "first_token":
                activation = raw_activation[layer_name][0]
            else:
                raise ValueError(
                    f"Unknown inv_lre.object_aggregation: {object_aggregation}"
                )
            activations_by_object[prompt.object_name].append(activation)
        return activations_by_object


@dataclass
class InvLreTrainingRunManager:
    """
    A helper class to train inverted LREs for a relation, keeping track of which LREs can be cached and re-used.
    A LRE for concept A can be reused for concept B if the prompts LRE A was trained on match the
    n_lre_non_concept_train_samples and n_lre_concept_train_samples requirements for LRE B as well.
    This class exists because training Inv LREs is expensive so we should avoid it unnecessarily
    """

    model: nn.Module
    tokenizer: Tokenizer
    hidden_layers_matcher: LayerMatcher
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
    _precomputed_inv_lres_by_object: dict[
        str, InvertedLinearRelationalEmbedding
    ] = field(init=False)

    def __post_init__(self) -> None:
        self._remaining_objects = set(self.prompts_by_object.keys())
        self._precomputed_inv_lres_by_object = {}

    def get_inv_lre_for_object(
        self, object_name: str
    ) -> InvertedLinearRelationalEmbedding:
        if object_name not in self._remaining_objects:
            raise ValueError(f"Object {object_name} has already been trained")
        self._remaining_objects.remove(object_name)
        if object_name in self._precomputed_inv_lres_by_object:
            logger.info(f"Using cached LRE for {object_name}")
            inv_lre = self._precomputed_inv_lres_by_object[object_name]
            del self._precomputed_inv_lres_by_object[object_name]
            return inv_lre
        return self._train_inv_lre_for_object(object_name)

    def _train_inv_lre_for_object(
        self, object_name: str
    ) -> InvertedLinearRelationalEmbedding:
        start_time = time()
        train_samples = self._get_train_samples_for_object(object_name)
        inv_lre = train_lre(
            self.model,
            self.tokenizer,
            self.hidden_layers_matcher,
            prompts=train_samples,
            relation_name=self.relation_name,
            subject_layer=self.subject_layer,
            object_layer=self.object_layer,
            object_aggregation=self.object_aggregation,
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
