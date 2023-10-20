from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

import torch
from tokenizers import Tokenizer

from multitoken_estimator.database import get_relation_to_lre_type_map
from multitoken_estimator.lib.logger import log_or_print
from multitoken_estimator.lib.token_utils import find_num_answer_tokens
from multitoken_estimator.lib.util import mean, mean_values
from multitoken_estimator.LinearRelationalEmbedding import LinearRelationalEmbedding
from multitoken_estimator.training.evaluate_accuracy_and_causality import (
    RelationAccuracyResult,
    avg_accuracy,
)
from multitoken_estimator.training.evaluate_relation_causality import (
    RelationCausalityResult,
    avg_causality,
)
from multitoken_estimator.training.LreTrainer import LreTrainer

CausalityTokenCountMethod = Literal["original", "target", "max_original_target"]


@dataclass
class TrainingStrategy:
    name: str
    run_fn: Callable[[], list[LinearRelationalEmbedding]]
    metadata: dict[str, Any]


Evaluator = Callable[
    [Sequence[LinearRelationalEmbedding]],
    tuple[dict[str, RelationAccuracyResult], dict[str, RelationCausalityResult]],
]


@dataclass
class BenchmarkResult:
    strategy_name: str
    lres: list[LinearRelationalEmbedding]
    relation_causality: dict[str, RelationCausalityResult]
    relation_accuracy: dict[str, RelationAccuracyResult]
    metadata: dict[str, Any]

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

    def accuracy_by_num_answer_tokens(self, tokenizer: Tokenizer) -> dict[int, float]:
        num_answer_tokens_to_accuracy: dict[int, list[float]] = defaultdict(list)
        for relation_accuracy in self.relation_accuracy.values():
            for result in relation_accuracy.prompt_eval_results:
                num_result_tokens = find_num_answer_tokens(
                    tokenizer,
                    base_prompt=result.prompt.text,
                    answer=result.prompt.answer,
                )
                num_answer_tokens_to_accuracy[num_result_tokens].append(
                    1.0 if result.is_correct else 0.0
                )

        return {
            num_answer_tokens: mean(accuracy)
            for num_answer_tokens, accuracy in num_answer_tokens_to_accuracy.items()
        }

    def total_by_num_accuracy_answer_tokens(
        self, tokenizer: Tokenizer
    ) -> dict[int, int]:
        num_answer_tokens_to_count: dict[int, int] = defaultdict(int)
        for relation_accuracy in self.relation_accuracy.values():
            for result in relation_accuracy.prompt_eval_results:
                num_result_tokens = find_num_answer_tokens(
                    tokenizer,
                    base_prompt=result.prompt.text,
                    answer=result.prompt.answer,
                )
                num_answer_tokens_to_count[num_result_tokens] += 1
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
        total_by_relation = self.total_by_relation
        causality_by_relation = self.causality_by_relation
        scores = []
        for relation, causality in causality_by_relation.items():
            if total_by_relation[relation] >= min_total_per_relation:
                scores.append(causality)
        return mean(scores)

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
        total_by_relation = self.total_by_relation
        causality_by_relation = self.causality_by_relation
        relation_to_lre_type = get_relation_to_lre_type_map()
        scores_by_lre_type: dict[str, list[float]] = defaultdict(list)
        for relation, causality in causality_by_relation.items():
            lre_type = relation_to_lre_type[relation]
            if total_by_relation[relation] >= min_total_per_relation:
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
        total_by_relation = self.total_by_relation
        accuracy_by_relation = self.accuracy_by_relation
        scores = []
        for relation, accuracy in accuracy_by_relation.items():
            if total_by_relation[relation] >= min_total_per_relation:
                scores.append(accuracy)
        return mean(scores)

    @property
    def accuracy_by_lre_type(self) -> dict[str, float]:
        return mean_values(
            [result.accuracy_by_lre_type for result in self.iteration_results]
        )

    def accuracy_by_lre_type_avg_over_relations(
        self, min_total_per_relation: int = 0
    ) -> dict[str, float]:
        # calculate accuracy by first finding accuracy per relation, and averaging relation scores
        # This means it doesn't matter if a relations has 1 test examples or 1000, they get the same weight
        total_by_relation = self.total_by_relation
        accuracy_by_relation = self.accuracy_by_relation
        relation_to_lre_type = get_relation_to_lre_type_map()
        scores_by_lre_type: dict[str, list[float]] = defaultdict(list)
        for relation, accuracy in accuracy_by_relation.items():
            lre_type = relation_to_lre_type[relation]
            if total_by_relation[relation] >= min_total_per_relation:
                scores_by_lre_type[lre_type].append(accuracy)
        return {
            lre_type: mean(scores) for lre_type, scores in scores_by_lre_type.items()
        }

    @property
    def accuracy_by_relation(self) -> dict[str, float]:
        return mean_values(
            [result.accuracy_by_relation for result in self.iteration_results]
        )

    @property
    def total_by_relation(self) -> dict[str, float]:
        """Total averaged over all iterations"""
        return mean_values(
            [result.total_by_relation for result in self.iteration_results]
        )

    def accuracy_by_num_answer_tokens(self, tokenizer: Tokenizer) -> dict[int, float]:
        return mean_values(
            [
                result.accuracy_by_num_answer_tokens(tokenizer)
                for result in self.iteration_results
            ]
        )

    def causality_by_num_answer_tokens(
        self,
        method: CausalityTokenCountMethod = "max_original_target",
    ) -> dict[int, float]:
        return mean_values(
            [
                result.causality_by_num_answer_tokens(method)
                for result in self.iteration_results
            ]
        )

    def total_by_num_accuracy_answer_tokens(
        self, tokenizer: Tokenizer
    ) -> dict[int, float]:
        """Total averaged over all iterations"""
        return mean_values(
            [
                result.total_by_num_accuracy_answer_tokens(tokenizer)
                for result in self.iteration_results
            ]
        )

    def total_by_num_causality_answer_tokens(
        self, method: CausalityTokenCountMethod = "max_original_target"
    ) -> dict[int, float]:
        """Total averaged over all iterations"""
        return mean_values(
            [
                result.total_by_num_causality_answer_tokens(method)
                for result in self.iteration_results
            ]
        )


def strategy_from_trainer(
    trainer: LreTrainer, strategy_name: str, kwargs: dict[str, Any]
) -> TrainingStrategy:
    """
    Create a training strategy from a trainer and a strategy name
    """

    def run_fn() -> list[LinearRelationalEmbedding]:
        return trainer.train_all_relations(**kwargs)

    return TrainingStrategy(strategy_name, run_fn, kwargs)


def benchmark_strategies(
    strategies: list[TrainingStrategy],
    evaluator: Evaluator,
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
    return results


def benchmark_strategy(
    strategy: TrainingStrategy, evaluator: Evaluator
) -> BenchmarkResult:
    """
    Run a training strategy and return the results
    """
    lres = strategy.run_fn()

    relation_accuracy, relation_causality = evaluator(lres)

    results = BenchmarkResult(
        strategy.name,
        lres,
        relation_accuracy=relation_accuracy,
        relation_causality=relation_causality,
        metadata=strategy.metadata,
    )
    return results
