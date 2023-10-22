from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional

import torch

from multitoken_estimator.data.data_loaders import get_relation_to_lre_type_map
from multitoken_estimator.lib.logger import log_or_print
from multitoken_estimator.lib.util import mean, mean_values
from multitoken_estimator.LinearRelationalEmbedding import LinearRelationalEmbedding
from multitoken_estimator.ObjectMappingModel import ObjectMappingModel
from multitoken_estimator.training.evaluate_accuracy_and_causality import (
    RelationAccuracyResult,
    avg_accuracy,
)
from multitoken_estimator.training.evaluate_relation_causality import (
    RelationCausalityResult,
    avg_causality,
)
from multitoken_estimator.training.LreEvaluator import LreEvaluator
from multitoken_estimator.training.LreTrainer import LresWithObjectMapping, LreTrainer

CausalityTokenCountMethod = Literal["original", "target", "max_original_target"]


@dataclass
class TrainingStrategy:
    name: str
    run_fn: Callable[[], LresWithObjectMapping]
    metadata: dict[str, Any]


@dataclass
class BenchmarkResult:
    strategy_name: str
    lres: list[LinearRelationalEmbedding]
    object_mapping: ObjectMappingModel
    relation_causality: dict[str, RelationCausalityResult]
    relation_accuracy: dict[str, RelationAccuracyResult]
    metadata: dict[str, Any]

    @property
    def relations(self) -> set[str]:
        return {lre.relation for lre in self.lres}

    @property
    def causality(self) -> float:
        return avg_causality(self.relation_causality.values())

    @property
    def causality_by_lre_type(self) -> dict[str, float]:
        relation_to_lre_type = get_relation_to_lre_type_map()
        lre_type_to_relation_causalities = defaultdict(list)
        for relation, lre_type in relation_to_lre_type.items():
            if relation in self.relation_causality:
                lre_type_to_relation_causalities[lre_type].append(
                    self.relation_causality[relation]
                )
        return {
            relation: avg_causality(relation_causalities)
            for relation, relation_causalities in lre_type_to_relation_causalities.items()
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
    def accuracy_by_lre_type(self) -> dict[str, float]:
        relation_to_lre_type = get_relation_to_lre_type_map()
        lre_type_to_relation_accuracies = defaultdict(list)
        for relation, lre_type in relation_to_lre_type.items():
            if relation in self.relation_accuracy:
                lre_type_to_relation_accuracies[lre_type].append(
                    self.relation_accuracy[relation]
                )
        return {
            lre_type: avg_accuracy(relation_accuracy)
            for lre_type, relation_accuracy in lre_type_to_relation_accuracies.items()
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
    def total_by_lre_type(self) -> dict[str, int]:
        relation_to_lre_type = get_relation_to_lre_type_map()
        lre_type_to_total: dict[str, int] = defaultdict(int)
        for relation, lre_type in relation_to_lre_type.items():
            if relation in self.relation_accuracy:
                lre_type_to_total[lre_type] += self.relation_accuracy[relation].total
        return lre_type_to_total


@dataclass
class BenchmarkIterationsResult:
    iteration_results: list[BenchmarkResult]

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
    def causality_by_lre_type(self) -> dict[str, float]:
        return mean_values(
            [result.causality_by_lre_type for result in self.iteration_results]
        )

    def causality_by_lre_type_avg_over_relations(
        self, min_total_per_relation: int = 0
    ) -> dict[str, float]:
        # calculate causality by first finding causality per relation, and averaging relation scores
        # This means it doesn't matter if a relations has 1 test examples or 1000, they get the same weight
        causality_by_relation = self._filter_metric_by_total(
            self.causality_by_relation, min_total_per_relation
        )
        relation_to_lre_type = get_relation_to_lre_type_map()
        scores_by_lre_type: dict[str, list[float]] = defaultdict(list)
        for relation, causality in causality_by_relation.items():
            lre_type = relation_to_lre_type[relation]
            scores_by_lre_type[lre_type].append(causality)
        return {
            lre_type: mean(scores) for lre_type, scores in scores_by_lre_type.items()
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
    def accuracy_by_lre_type(self) -> dict[str, float]:
        return mean_values(
            [result.accuracy_by_lre_type for result in self.iteration_results]
        )

    # TODO: this doesn't belong in the base class, since this assumes we're using the LRE dataset
    def accuracy_by_lre_type_avg_over_relations(
        self, min_total_per_relation: int = 0
    ) -> dict[str, float]:
        # calculate accuracy by first finding accuracy per relation, and averaging relation scores
        # This means it doesn't matter if a relations has 1 test examples or 1000, they get the same weight
        accuracy_by_relation = self._filter_metric_by_total(
            self.accuracy_by_relation, min_total_per_relation
        )
        relation_to_lre_type = get_relation_to_lre_type_map()
        scores_by_lre_type: dict[str, list[float]] = defaultdict(list)
        for relation, accuracy in accuracy_by_relation.items():
            lre_type = relation_to_lre_type[relation]
            scores_by_lre_type[lre_type].append(accuracy)
        return {
            lre_type: mean(scores) for lre_type, scores in scores_by_lre_type.items()
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


def strategy_from_trainer(
    trainer: LreTrainer, strategy_name: str, kwargs: dict[str, Any]
) -> TrainingStrategy:
    """
    Create a training strategy from a trainer and a strategy name
    """

    def run_fn() -> LresWithObjectMapping:
        return trainer.train_all_relations_with_object_mapping(**kwargs)

    return TrainingStrategy(strategy_name, run_fn, kwargs)


def benchmark_strategies(
    strategies: list[TrainingStrategy],
    evaluator: LreEvaluator,
    save_progress_path: Optional[str | Path] = None,
    force_rerun: bool = False,
    verbose: bool = True,
) -> dict[str, BenchmarkResult]:
    """
    Run a list of training strategies and return the results
    """
    results = {}
    if (
        save_progress_path is not None
        and os.path.exists(save_progress_path)
        and not force_rerun
    ):
        results = torch.load(save_progress_path)
        log_or_print(
            f"Loaded {len(results)} results from {save_progress_path}",
            verbose,
        )
    for strategy in strategies:
        if strategy.name in results:
            log_or_print(
                f"Skipping training strategy {strategy.name}, already run",
                verbose,
            )
        else:
            log_or_print(
                f"Running training strategy {strategy.name}",
                verbose,
            )
            results[strategy.name] = benchmark_strategy(strategy, evaluator)
        if save_progress_path is not None:
            torch.save(results, save_progress_path)
        if verbose:
            print_benchmark_result(results[strategy.name])
    return results


def benchmark_strategy(
    strategy: TrainingStrategy, evaluator: LreEvaluator, verbose: bool = True
) -> BenchmarkResult:
    """
    Run a training strategy and return the results
    """
    lres_with_mapping = strategy.run_fn()

    relation_accuracy = evaluator.evaluate_accuracy(lres_with_mapping, verbose=verbose)
    relation_causality = evaluator.evaluate_causality(
        lres_with_mapping, verbose=verbose
    )

    results = BenchmarkResult(
        strategy.name,
        lres_with_mapping.lres,
        lres_with_mapping.object_mapping,
        relation_accuracy=relation_accuracy,
        relation_causality=relation_causality,
        metadata=strategy.metadata,
    )
    return results


def print_benchmark_result(result: BenchmarkResult) -> None:
    """
    Print a benchmark result
    """
    print(f"Strategy: {result.strategy_name}")
    print(f"accuracy: {result.accuracy:.3f}, causality: {result.causality:.3f}")
    print("accuracy by LRE type")
    for lre_type, accuracy in result.accuracy_by_lre_type.items():
        print(f"\t{lre_type}: {accuracy:.3f}")
    print("causality by LRE type")
    for lre_type, causality in result.causality_by_lre_type.items():
        print(f"\t{lre_type}: {causality:.3f}")
    print("accuracy per num answer tokens")
    for ntoks, accuracy in result.accuracy_by_num_answer_tokens.items():
        print(f"\t{ntoks}: {accuracy:.3f}")
    print("causality per num answer tokens")
    for ntoks, causality in result.causality_by_num_answer_tokens().items():
        print(f"\t{ntoks}: {causality:.3f}")
    print("total per num accuracy answer tokens")
    for ntoks, total in result.total_by_num_accuracy_answer_tokens.items():
        print(f"\t{ntoks}: {total:.3f}")
    print("total per num causality answer tokens")
    for ntoks, total in result.total_by_num_causality_answer_tokens().items():
        print(f"\t{ntoks}: {total:.3f}")
