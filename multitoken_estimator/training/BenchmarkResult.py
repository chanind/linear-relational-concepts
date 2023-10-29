from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from multitoken_estimator.Concept import Concept
from multitoken_estimator.evaluation.evaluate_accuracy_and_causality import (
    RelationAccuracyResult,
    avg_accuracy,
)
from multitoken_estimator.evaluation.evaluate_relation_causality import (
    RelationCausalityResult,
    avg_causality,
)
from multitoken_estimator.lib.util import mean

CausalityTokenCountMethod = Literal["original", "target", "max_original_target"]


@dataclass
class BenchmarkResult:
    strategy_name: str
    concepts: list[Concept]
    relation_causality: dict[str, RelationCausalityResult]
    relation_accuracy: dict[str, RelationAccuracyResult]
    relation_category_map: dict[str, str]
    seed: int | float | str
    metadata: dict[str, Any]

    @property
    def relations(self) -> set[str]:
        return {concept.relation for concept in self.concepts}

    @property
    def causality(self) -> float:
        return avg_causality(self.relation_causality.values())

    @property
    def causality_by_category(self) -> dict[str, float]:
        category_to_relation_causalities = defaultdict(list)
        for relation, category in self.relation_category_map.items():
            if relation in self.relation_causality:
                category_to_relation_causalities[category].append(
                    self.relation_causality[relation]
                )
        return {
            relation: avg_causality(relation_causalities)
            for relation, relation_causalities in category_to_relation_causalities.items()
        }

    @property
    def causality_by_relation(self) -> dict[str, float]:
        return {
            relation: relation_causality.causality
            for relation, relation_causality in self.relation_causality.items()
        }

    def causality_by_num_answer_tokens(
        self,
        method: CausalityTokenCountMethod = "max_original_target",
    ) -> dict[int, float]:
        num_answer_tokens_to_causalities: dict[int, list[float]] = defaultdict(list)
        for relation_causality in self.relation_causality.values():
            for result in relation_causality.prompt_edit_results:
                num_orig_tokens = result.num_original_tokens
                num_target_tokens = result.num_target_tokens
                if method == "original":
                    num_result_tokens = num_orig_tokens
                elif method == "target":
                    num_result_tokens = num_target_tokens
                elif method == "max_original_target":
                    num_result_tokens = max(num_orig_tokens, num_target_tokens)
                else:
                    raise ValueError(f"Unknown method {method}")
                num_answer_tokens_to_causalities[num_result_tokens].append(
                    1.0 if result.is_succesful else 0.0
                )

        return {
            num_answer_tokens: mean(causalities)
            for num_answer_tokens, causalities in num_answer_tokens_to_causalities.items()
        }

    @property
    def accuracy(self) -> float:
        return avg_accuracy(self.relation_accuracy.values())

    @property
    def accuracy_by_category(self) -> dict[str, float]:
        category_to_relation_accuracies = defaultdict(list)
        for relation, category in self.relation_category_map.items():
            if relation in self.relation_accuracy:
                category_to_relation_accuracies[category].append(
                    self.relation_accuracy[relation]
                )
        return {
            category: avg_accuracy(relation_accuracy)
            for category, relation_accuracy in category_to_relation_accuracies.items()
        }

    @property
    def accuracy_by_relation(self) -> dict[str, float]:
        return {
            relation: relation_accuracy.accuracy
            for relation, relation_accuracy in self.relation_accuracy.items()
        }

    @property
    def accuracy_by_num_answer_tokens(self) -> dict[int, float]:
        num_answer_tokens_to_accuracy: dict[int, list[float]] = defaultdict(list)
        for relation_accuracy in self.relation_accuracy.values():
            for result in relation_accuracy.prompt_eval_results:
                num_answer_tokens_to_accuracy[result.num_answer_tokens].append(
                    1.0 if result.is_correct else 0.0
                )

        return {
            num_answer_tokens: mean(accuracy)
            for num_answer_tokens, accuracy in num_answer_tokens_to_accuracy.items()
        }

    @property
    def total_by_num_accuracy_answer_tokens(self) -> dict[int, int]:
        num_answer_tokens_to_count: dict[int, int] = defaultdict(int)
        for relation_accuracy in self.relation_accuracy.values():
            for result in relation_accuracy.prompt_eval_results:
                num_answer_tokens_to_count[result.num_answer_tokens] += 1
        return num_answer_tokens_to_count

    def total_by_num_causality_answer_tokens(
        self,
        method: CausalityTokenCountMethod = "max_original_target",
    ) -> dict[int, int]:
        num_answer_tokens_to_count: dict[int, int] = defaultdict(int)
        for relation_causality in self.relation_causality.values():
            for result in relation_causality.prompt_edit_results:
                num_orig_tokens = result.num_original_tokens
                num_target_tokens = result.num_target_tokens
                if method == "original":
                    num_result_tokens = num_orig_tokens
                elif method == "target":
                    num_result_tokens = num_target_tokens
                elif method == "max_original_target":
                    num_result_tokens = max(num_orig_tokens, num_target_tokens)
                else:
                    raise ValueError(f"Unknown method {method}")
                num_answer_tokens_to_count[num_result_tokens] += 1
        return num_answer_tokens_to_count

    @property
    def total_by_relation(self) -> dict[str, int]:
        return {
            relation: relation_accuracy.total
            for relation, relation_accuracy in self.relation_accuracy.items()
        }

    @property
    def total_by_category(self) -> dict[str, int]:
        category_to_total: dict[str, int] = defaultdict(int)
        for relation, category in self.relation_category_map.items():
            if relation in self.relation_accuracy:
                category_to_total[category] += self.relation_accuracy[relation].total
        return category_to_total
