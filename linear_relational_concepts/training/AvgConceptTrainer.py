from collections import defaultdict
from dataclasses import asdict, dataclass

import torch
from typing_extensions import override

from linear_relational_concepts.Concept import Concept
from linear_relational_concepts.lib.extract_token_activations import (
    extract_token_activations,
)
from linear_relational_concepts.lib.layer_matching import get_layer_name
from linear_relational_concepts.lib.token_utils import find_final_word_token_index
from linear_relational_concepts.lib.torch_utils import get_device
from linear_relational_concepts.PromptGenerator import AUGMENTATION_MODIFIERS, Prompt
from linear_relational_concepts.training.ConceptTrainer import (
    ConceptTrainer,
    ConceptTrainerOptions,
)
from linear_relational_concepts.training.train_avg_concept_vector import (
    train_avg_concept_vector,
)


@dataclass
class AvgConceptTrainerOptions(ConceptTrainerOptions):
    augment_prompts: bool = False
    filter_training_prompts: bool = True
    exclude_fsl_examples_of_object: bool = False
    n_fsl_prompts: int = 4
    batch_size: int = 8
    seed: int | float | str = 42


class AvgConceptTrainer(ConceptTrainer[AvgConceptTrainerOptions]):
    @override
    def train_relation_concepts(
        self,
        relation: str,
        opts: AvgConceptTrainerOptions,
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
        opts: AvgConceptTrainerOptions,
    ) -> Concept:
        pos_samples = subject_activations_by_object[object]
        vector, metadata = train_avg_concept_vector(
            pos_samples,
        )

        return Concept(
            object=object,
            relation=relation,
            vector=vector,
            layer=layer,
            metadata={
                "opts": asdict(opts),
                **metadata,
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
