from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from typing_extensions import override

from linear_relational_concepts.Concept import Concept
from linear_relational_concepts.lib.extract_token_activations import (
    extract_token_activations,
)
from linear_relational_concepts.lib.layer_matching import get_layer_name
from linear_relational_concepts.lib.token_utils import find_final_word_token_index
from linear_relational_concepts.lib.torch_utils import get_device
from linear_relational_concepts.lib.util import stable_shuffle
from linear_relational_concepts.PromptGenerator import AUGMENTATION_MODIFIERS, Prompt
from linear_relational_concepts.training.ConceptTrainer import (
    ConceptTrainer,
    ConceptTrainerOptions,
)
from linear_relational_concepts.training.train_concept_vector_via_svm import (
    train_concept_vector_via_svm,
)


@dataclass
class SvmConceptTrainerOptions(ConceptTrainerOptions):
    max_pos_training_activations: Optional[int] = None
    max_neg_training_activations: Optional[int] = None
    fit_intercept: bool = False
    augment_prompts: bool = False
    filter_training_prompts: bool = True
    exclude_fsl_examples_of_object: bool = False
    n_fsl_prompts: int = 4
    batch_size: int = 8
    max_iter: int = 1000
    seed: int | float | str = 42


class SvmConceptTrainer(ConceptTrainer[SvmConceptTrainerOptions]):
    @override
    def train_relation_concepts(
        self,
        relation: str,
        opts: SvmConceptTrainerOptions,
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
        subject_activations_by_object = self._extract_subject_activations_by_object(
            prompts=relation_prompts,
            batch_size=opts.batch_size,
            layer=opts.layer,
            show_progress=verbose,
            move_to_cpu=True,
        )
        concepts = [
            self._train_concept(
                relation=relation,
                object=object,
                layer=opts.layer,
                subject_activations_by_object=subject_activations_by_object,
                opts=opts,
            )
            for object in subject_activations_by_object.keys()
        ]
        return concepts

    def _train_concept(
        self,
        relation: str,
        object: str,
        layer: int,
        subject_activations_by_object: dict[str, list[torch.Tensor]],
        opts: SvmConceptTrainerOptions,
    ) -> Concept:
        pos_samples = subject_activations_by_object[object]
        neg_samples: list[torch.Tensor] = []
        for other_object, other_samples in subject_activations_by_object.items():
            if other_object != object:
                neg_samples.extend(other_samples)
        if opts.max_pos_training_activations is not None:
            shuffled_pos_samples = stable_shuffle(
                pos_samples, seed=f"{relation}-{object}-{opts.seed}"
            )
            pos_samples = shuffled_pos_samples[: opts.max_pos_training_activations]
        if opts.max_neg_training_activations is not None:
            shuffled_neg_samples = stable_shuffle(
                neg_samples, seed=f"{relation}-{object}-{opts.seed}"
            )
            neg_samples = shuffled_neg_samples[: opts.max_neg_training_activations]
        vector = train_concept_vector_via_svm(
            pos_activations=pos_samples,
            neg_activations=neg_samples,
            max_iter=opts.max_iter,
            fit_intercept=opts.fit_intercept,
        )

        return Concept(
            object=object,
            relation=relation,
            vector=vector,
            layer=layer,
            metadata={
                "num_pos_activations": len(pos_samples),
                "num_neg_activations": len(neg_samples),
                "opts": asdict(opts),
            },
        )

    def _extract_subject_activations_by_object(
        self,
        prompts: list[Prompt],
        batch_size: int,
        layer: int,
        show_progress: bool = True,
        move_to_cpu: bool = True,
    ) -> dict[str, list[torch.Tensor]]:
        activations_by_object: dict[str, list[torch.Tensor]] = defaultdict(list)
        final_subj_tok_indices = [
            find_final_word_token_index(self.tokenizer, prompt.text, prompt.subject)
            for prompt in prompts
        ]

        layer_name = get_layer_name(self.model, self.layer_matcher, layer)
        raw_activations = extract_token_activations(
            self.model,
            self.tokenizer,
            layers=[layer_name],
            texts=[promt.text for promt in prompts],
            token_indices=final_subj_tok_indices,
            device=get_device(self.model),
            batch_size=batch_size,
            show_progress=show_progress,
            move_results_to_cpu=move_to_cpu,
        )
        for prompt, raw_activation in zip(prompts, raw_activations):
            activations_by_object[prompt.object_name].append(
                raw_activation[layer_name][0]
            )
        return activations_by_object
