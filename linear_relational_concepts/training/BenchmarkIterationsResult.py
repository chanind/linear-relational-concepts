from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import stdev
from typing import Any, Callable, Mapping, Sequence

from linear_relational_concepts.lib.util import mean, mean_values
from linear_relational_concepts.training.BenchmarkResult import (
    BenchmarkResult,
    CausalityTokenCountMethod,
)


@dataclass
class AggregateMetric:
    values: list[float]

    @property
    def mean(self) -> float:
        return mean(self.values)

    @property
    def stdev(self) -> float:
        if len(self.values) <= 1:
            return 0
        return stdev(self.values)

    def __repr__(self) -> str:
        return f"{self.mean:.3f} ± {self.stdev:.3f}"


def _agg_values(items: Sequence[Mapping[str, float]]) -> dict[str, AggregateMetric]:
    """
    return a dict of AggregateMetrics from lists of dicts of numbers
    """
    return {
        key: AggregateMetric([item[key] for item in items if key in item])
        for key in set(key for item in items for key in item.keys())
    }


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
    def causality(self) -> AggregateMetric:
        return AggregateMetric([result.causality for result in self.iteration_results])

    def multitoken_causality(
        self, method: CausalityTokenCountMethod = "max_original_target"
    ) -> AggregateMetric:
        return AggregateMetric(
            [result.multitoken_causality(method) for result in self.iteration_results]
        )

    def single_token_causality(
        self, method: CausalityTokenCountMethod = "max_original_target"
    ) -> AggregateMetric:
        return AggregateMetric(
            [result.single_token_causality(method) for result in self.iteration_results]
        )

    def causality_avg_over_relations(
        self, min_total_per_relation: int = 0
    ) -> AggregateMetric:
        # calculate causality by first finding causality per relation, and averaging relation scores
        # This means it doesn't matter if a relations has 1 test examples or 1000, they get the same weight
        causality_by_relation = self._filter_metric_by_total(
            self.causality_by_relation, min_total_per_relation
        )
        return AggregateMetric(
            [causality.mean for causality in causality_by_relation.values()]
        )

    @property
    def causality_by_category(self) -> dict[str, float]:
        return mean_values(
            [result.causality_by_category for result in self.iteration_results]
        )

    def causality_by_category_avg_over_relations(
        self, min_total_per_relation: int = 0
    ) -> dict[str, AggregateMetric]:
        # calculate causality by first finding causality per relation, and averaging relation scores
        # This means it doesn't matter if a relations has 1 test examples or 1000, they get the same weight
        causality_by_relation = self._filter_metric_by_total(
            self.causality_by_relation, min_total_per_relation
        )
        scores_by_category: dict[str, list[AggregateMetric]] = defaultdict(list)
        for relation, causality in causality_by_relation.items():
            if relation in self.relation_category_map:
                category = self.relation_category_map[relation]
                scores_by_category[category].append(causality)
        return {
            category: AggregateMetric([score.mean for score in scores])
            for category, scores in scores_by_category.items()
        }

    @property
    def causality_by_relation(self) -> dict[str, AggregateMetric]:
        return _agg_values(
            [result.causality_by_relation for result in self.iteration_results]
        )

    @property
    def accuracy(self) -> AggregateMetric:
        return AggregateMetric([result.accuracy for result in self.iteration_results])

    @property
    def multitoken_accuracy(self) -> AggregateMetric:
        return AggregateMetric(
            [result.multitoken_accuracy for result in self.iteration_results]
        )

    @property
    def single_token_accuracy(self) -> AggregateMetric:
        return AggregateMetric(
            [result.single_token_accuracy for result in self.iteration_results]
        )

    def accuracy_avg_over_relations(
        self, min_total_per_relation: int = 0
    ) -> AggregateMetric:
        # calculate accuracy by first finding accuracy per relation, and averaging relation scores
        # This means it doesn't matter if a relations has 1 test examples or 1000, they get the same weight
        accuracy_by_relation = self._filter_metric_by_total(
            self.accuracy_by_relation, min_total_per_relation
        )
        return AggregateMetric(
            [accuracy.mean for accuracy in accuracy_by_relation.values()]
        )

    @property
    def accuracy_by_category(self) -> dict[str, AggregateMetric]:
        return _agg_values(
            [result.accuracy_by_category for result in self.iteration_results]
        )

    def accuracy_by_category_avg_over_relations(
        self, min_total_per_relation: int = 0
    ) -> dict[str, AggregateMetric]:
        # calculate accuracy by first finding accuracy per relation, and averaging relation scores
        # This means it doesn't matter if a relations has 1 test examples or 1000, they get the same weight
        accuracy_by_relation = self._filter_metric_by_total(
            self.accuracy_by_relation, min_total_per_relation
        )
        scores_by_category: dict[str, list[AggregateMetric]] = defaultdict(list)
        for relation, accuracy in accuracy_by_relation.items():
            if relation in self.relation_category_map:
                category = self.relation_category_map[relation]
                scores_by_category[category].append(accuracy)
        return {
            category: AggregateMetric([score.mean for score in scores])
            for category, scores in scores_by_category.items()
        }

    def _filter_metric_by_total(
        self,
        metric_per_relation: Mapping[str, AggregateMetric],
        min_total_per_relation: int = 0,
    ) -> dict[str, AggregateMetric]:
        total_by_relation = self.total_by_relation
        return {
            relation: metric
            for relation, metric in metric_per_relation.items()
            if total_by_relation[relation].mean >= min_total_per_relation
        }

    @property
    def accuracy_by_relation(self) -> dict[str, AggregateMetric]:
        return self._agg_vals(lambda res: res.accuracy_by_relation)

    @property
    def total_by_relation(self) -> dict[str, AggregateMetric]:
        """Total averaged over all iterations"""
        return self._agg_vals(lambda res: res.total_by_relation)

    @property
    def accuracy_by_num_answer_tokens(self) -> dict[int, AggregateMetric]:
        return self._agg_vals(lambda res: res.accuracy_by_num_answer_tokens)

    def causality_by_num_answer_tokens(
        self,
        method: CausalityTokenCountMethod = "max_original_target",
    ) -> dict[int, AggregateMetric]:
        return self._agg_vals(lambda res: res.causality_by_num_answer_tokens(method))

    @property
    def total_by_num_accuracy_answer_tokens(self) -> dict[int, AggregateMetric]:
        """Total averaged over all iterations"""
        return self._agg_vals(lambda res: res.total_by_num_accuracy_answer_tokens)

    def total_by_num_causality_answer_tokens(
        self, method: CausalityTokenCountMethod = "max_original_target"
    ) -> dict[int, AggregateMetric]:
        """Total averaged over all iterations"""
        return self._agg_vals(
            lambda res: res.total_by_num_causality_answer_tokens(method)
        )

    def _agg_vals(
        self, result_cb: Callable[[BenchmarkResult], Mapping[Any, float]]
    ) -> dict[Any, AggregateMetric]:
        return _agg_values([result_cb(result) for result in self.iteration_results])


def print_iterations_result(
    result: BenchmarkIterationsResult, min_total_per_relation: int = 1
) -> None:
    print(f"accuracy: {result.accuracy}, causality: {result.causality}")
    print(
        f"accuracy (avg relations): {result.accuracy_avg_over_relations(min_total_per_relation)}, causality (avg relations): {result.causality_avg_over_relations(min_total_per_relation)}"
    )
    print("accuracy per LRE type")
    for category, accuracy in result.accuracy_by_category_avg_over_relations(
        min_total_per_relation
    ).items():
        print(f"\t{category}: {accuracy}")
    print("causality per LRE type")
    for category, causality in result.causality_by_category_avg_over_relations(
        5
    ).items():
        print(f"\t{category}: {causality}")
    print("accuracy per num answer tokens")
    for ntoks, accuracy in result.accuracy_by_num_answer_tokens.items():
        print(f"\t{ntoks}: {accuracy}")
    print("causality per num answer tokens")
    for ntoks, causality in result.causality_by_num_answer_tokens().items():
        print(f"\t{ntoks}: {causality}")
    print("total per num accuracy answer tokens")
    for ntoks, total in result.total_by_num_accuracy_answer_tokens.items():
        print(f"\t{ntoks}: {total}")
    print("total per num causality answer tokens")
    for ntoks, total in result.total_by_num_causality_answer_tokens().items():
        print(f"\t{ntoks}: {total}")
    print("\n")
