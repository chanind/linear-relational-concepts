from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from multitoken_estimator.lib.util import mean, mean_values
from multitoken_estimator.training.BenchmarkResult import (
    BenchmarkResult,
    CausalityTokenCountMethod,
)


@dataclass
class BenchmarkIterationsResult:
    iteration_results: list[BenchmarkResult]

    @property
    def relation_category_map(self) -> dict[str, str]:
        mapping = {}
        for result in self.iteration_results:
            mapping.update(result.relation_category_map)
        return mapping

    @property
    def causality(self) -> float:
        return mean([result.causality for result in self.iteration_results])

    def causality_avg_over_relations(self, min_total_per_relation: int = 0) -> float:
        # calculate causality by first finding causality per relation, and averaging relation scores
        # This means it doesn't matter if a relations has 1 test examples or 1000, they get the same weight
        causality_by_relation = self._filter_metric_by_total(
            self.causality_by_relation, min_total_per_relation
        )
        return mean([causality for causality in causality_by_relation.values()])

    @property
    def causality_by_category(self) -> dict[str, float]:
        return mean_values(
            [result.causality_by_category for result in self.iteration_results]
        )

    def causality_by_category_avg_over_relations(
        self, min_total_per_relation: int = 0
    ) -> dict[str, float]:
        # calculate causality by first finding causality per relation, and averaging relation scores
        # This means it doesn't matter if a relations has 1 test examples or 1000, they get the same weight
        causality_by_relation = self._filter_metric_by_total(
            self.causality_by_relation, min_total_per_relation
        )
        scores_by_category: dict[str, list[float]] = defaultdict(list)
        for relation, causality in causality_by_relation.items():
            if relation in self.relation_category_map:
                category = self.relation_category_map[relation]
                scores_by_category[category].append(causality)
        return {
            category: mean(scores) for category, scores in scores_by_category.items()
        }

    @property
    def causality_by_relation(self) -> dict[str, float]:
        return mean_values(
            [result.causality_by_relation for result in self.iteration_results]
        )

    @property
    def accuracy(self) -> float:
        return mean([result.accuracy for result in self.iteration_results])

    def accuracy_avg_over_relations(self, min_total_per_relation: int = 0) -> float:
        # calculate accuracy by first finding accuracy per relation, and averaging relation scores
        # This means it doesn't matter if a relations has 1 test examples or 1000, they get the same weight
        accuracy_by_relation = self._filter_metric_by_total(
            self.accuracy_by_relation, min_total_per_relation
        )
        return mean([accuracy for accuracy in accuracy_by_relation.values()])

    @property
    def accuracy_by_category(self) -> dict[str, float]:
        return mean_values(
            [result.accuracy_by_category for result in self.iteration_results]
        )

    def accuracy_by_category_avg_over_relations(
        self, min_total_per_relation: int = 0
    ) -> dict[str, float]:
        # calculate accuracy by first finding accuracy per relation, and averaging relation scores
        # This means it doesn't matter if a relations has 1 test examples or 1000, they get the same weight
        accuracy_by_relation = self._filter_metric_by_total(
            self.accuracy_by_relation, min_total_per_relation
        )
        scores_by_category: dict[str, list[float]] = defaultdict(list)
        for relation, accuracy in accuracy_by_relation.items():
            if relation in self.relation_category_map:
                category = self.relation_category_map[relation]
                scores_by_category[category].append(accuracy)
        return {
            category: mean(scores) for category, scores in scores_by_category.items()
        }

    def _filter_metric_by_total(
        self, metric_per_relation: Mapping[str, float], min_total_per_relation: int = 0
    ) -> dict[str, float]:
        total_by_relation = self.total_by_relation
        return {
            relation: metric
            for relation, metric in metric_per_relation.items()
            if total_by_relation[relation] >= min_total_per_relation
        }

    @property
    def accuracy_by_relation(self) -> dict[str, float]:
        return self._mean_vals(lambda res: res.accuracy_by_relation)

    @property
    def total_by_relation(self) -> dict[str, float]:
        """Total averaged over all iterations"""
        return self._mean_vals(lambda res: res.total_by_relation)

    @property
    def accuracy_by_num_answer_tokens(self) -> dict[int, float]:
        return self._mean_vals(lambda res: res.accuracy_by_num_answer_tokens)

    def causality_by_num_answer_tokens(
        self,
        method: CausalityTokenCountMethod = "max_original_target",
    ) -> dict[int, float]:
        return self._mean_vals(lambda res: res.causality_by_num_answer_tokens(method))

    @property
    def total_by_num_accuracy_answer_tokens(self) -> dict[int, float]:
        """Total averaged over all iterations"""
        return self._mean_vals(lambda res: res.total_by_num_accuracy_answer_tokens)

    def total_by_num_causality_answer_tokens(
        self, method: CausalityTokenCountMethod = "max_original_target"
    ) -> dict[int, float]:
        """Total averaged over all iterations"""
        return self._mean_vals(
            lambda res: res.total_by_num_causality_answer_tokens(method)
        )

    def _mean_vals(
        self, result_cb: Callable[[BenchmarkResult], Mapping[Any, float]]
    ) -> dict[Any, float]:
        return mean_values([result_cb(result) for result in self.iteration_results])
