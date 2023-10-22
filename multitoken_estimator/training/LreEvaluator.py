from dataclasses import dataclass

from tokenizers import Tokenizer
from torch import nn

from multitoken_estimator.data.database import Database
from multitoken_estimator.lib.layer_matching import LayerMatcher
from multitoken_estimator.training.evaluate_accuracy_and_causality import (
    RelationAccuracyResult,
    evaluate_causality,
    evaluate_relation_classification_accuracy,
)
from multitoken_estimator.training.evaluate_relation_causality import (
    RelationCausalityResult,
)
from multitoken_estimator.training.LreTrainer import LresWithObjectMapping


@dataclass
class LreEvaluator:
    model: nn.Module
    tokenizer: Tokenizer
    layer_matcher: LayerMatcher
    dataset: Database
    batch_size: int = 8
    inv_lre_rank: int = 100
    use_zs_prompts: bool = False
    causality_magnitude_multiplier: float = 1.0
    causality_edit_single_layer_only: bool = True
    causality_use_remove_concept_projection_magnitude: bool = False

    def evaluate_causality(
        self, lres_with_mapping: LresWithObjectMapping, verbose: bool = True
    ) -> dict[str, RelationCausalityResult]:
        causality_results = evaluate_causality(
            self.model,
            self.tokenizer,
            self.layer_matcher,
            lres=lres_with_mapping.lres,
            object_mapping=lres_with_mapping.object_mapping,
            dataset=self.dataset,
            batch_size=self.batch_size,
            inv_lre_rank=self.inv_lre_rank,
            verbose=verbose,
            use_zs_prompts=self.use_zs_prompts,
            magnitude_multiplier=self.causality_magnitude_multiplier,
            edit_single_layer_only=self.causality_edit_single_layer_only,
            use_remove_concept_projection_magnitude=self.causality_use_remove_concept_projection_magnitude,
        )
        return {result.relation: result for result in causality_results}

    def evaluate_accuracy(
        self, lres_with_mapping: LresWithObjectMapping, verbose: bool = True
    ) -> dict[str, RelationAccuracyResult]:
        accuracy_results = evaluate_relation_classification_accuracy(
            self.model,
            self.tokenizer,
            self.layer_matcher,
            lres=lres_with_mapping.lres,
            object_mapping=lres_with_mapping.object_mapping,
            dataset=self.dataset,
            batch_size=self.batch_size,
            inv_lre_rank=self.inv_lre_rank,
            verbose=verbose,
            use_zs_prompts=self.use_zs_prompts,
        )
        return {result.relation: result for result in accuracy_results}
