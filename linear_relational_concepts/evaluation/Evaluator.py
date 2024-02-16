from dataclasses import dataclass, field, replace

from linear_relational import Concept, LayerMatcher, PromptValidator
from tokenizers import Tokenizer
from torch import nn

from linear_relational_concepts.data.RelationDataset import Relation, RelationDataset
from linear_relational_concepts.PromptGenerator import PromptGenerator

from .evaluate_accuracy_and_causality import (
    RelationAccuracyResult,
    evaluate_causality,
    evaluate_relation_classification_accuracy,
)
from .evaluate_relation_causality import RelationCausalityResult


@dataclass
class Evaluator:
    model: nn.Module
    tokenizer: Tokenizer
    layer_matcher: LayerMatcher
    dataset: RelationDataset
    prompt_validator: PromptValidator
    batch_size: int = 8
    use_zs_prompts: bool = False
    causality_magnitude_multiplier: float = 1.0
    causality_edit_single_layer_only: bool = True
    causality_use_remove_concept_projection_magnitude: bool = False
    seed: int | float | str = 42

    prompt_generator: PromptGenerator = field(init=False)

    def __post_init__(self) -> None:
        self.prompt_generator = PromptGenerator(self.reformulated_dataset, self.seed)

    @property
    def reformulated_dataset(self) -> RelationDataset:
        """Return new concepts data replacing prompts with zs_prompts if self.use_zs_prompts is True"""
        if not self.use_zs_prompts:
            return self.dataset

        def reformulate_relation(relation: Relation) -> Relation:
            if relation.zs_templates is None:
                raise ValueError(f"Missing zs_prompts in sample data for {relation}")
            return replace(
                relation,
                templates=relation.zs_templates,
            )

        return self.dataset.reformulate_relations(reformulate_relation)

    def evaluate_causality(
        self, concepts: list[Concept], verbose: bool = True
    ) -> dict[str, RelationCausalityResult]:
        causality_results = evaluate_causality(
            self.model,
            self.tokenizer,
            self.layer_matcher,
            concepts=concepts,
            prompt_generator=self.prompt_generator,
            batch_size=self.batch_size,
            verbose=verbose,
            magnitude_multiplier=self.causality_magnitude_multiplier,
            edit_single_layer_only=self.causality_edit_single_layer_only,
            use_remove_concept_projection_magnitude=self.causality_use_remove_concept_projection_magnitude,
            prompt_validator=self.prompt_validator,
        )
        return {result.relation: result for result in causality_results}

    def evaluate_accuracy(
        self, concepts: list[Concept], verbose: bool = True
    ) -> dict[str, RelationAccuracyResult]:
        accuracy_results = evaluate_relation_classification_accuracy(
            self.model,
            self.tokenizer,
            self.layer_matcher,
            concepts=concepts,
            prompt_generator=self.prompt_generator,
            batch_size=self.batch_size,
            verbose=verbose,
            prompt_validator=self.prompt_validator,
        )
        return {result.relation: result for result in accuracy_results}

    def find_num_valid_prompts_per_relation(
        self, show_progress: bool = True
    ) -> dict[str, int]:
        prompts_by_relation = self.prompt_generator.generate_prompts_for_all_relations(
            num_fsl_examples=0
        )
        valid_prompts_by_relation = {
            relation: self.prompt_validator.filter_prompts(
                prompts, batch_size=self.batch_size, show_progress=show_progress
            )
            for relation, prompts in prompts_by_relation.items()
        }
        return {
            relation.name: len(valid_prompts_by_relation.get(relation.name, []))
            for relation in self.dataset.relations
        }

    def filter_relations_with_few_eval_prompts(
        self,
        dataset: RelationDataset,
        min_relation_eval_prompts: int,
        show_progress: bool = True,
    ) -> RelationDataset:
        """
        Filter a dataset so it only has relations for which this evaluator has
        sufficient prompts to evaluate. This is useful to avoid training relation
        concepts for which we don't have the ability to evaluate properly
        """
        num_valid_prompts_per_relation = self.find_num_valid_prompts_per_relation(
            show_progress=show_progress
        )
        valid_relation_names = {
            relation
            for relation, num_prompts in num_valid_prompts_per_relation.items()
            if num_prompts >= min_relation_eval_prompts
        }
        filtered_dataset = RelationDataset()
        for relation in dataset.relations:
            if relation.name in valid_relation_names:
                filtered_dataset.add_relation(relation)
        for sample in dataset.samples:
            if sample.relation in valid_relation_names:
                filtered_dataset.add_sample(sample)
        return filtered_dataset
