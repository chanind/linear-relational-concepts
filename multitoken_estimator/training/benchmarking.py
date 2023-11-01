from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time
from typing import Any, Callable, Optional, TypeVar

import torch

from multitoken_estimator.Concept import Concept
from multitoken_estimator.evaluation.Evaluator import Evaluator
from multitoken_estimator.lib.logger import log_or_print
from multitoken_estimator.training.BenchmarkResult import BenchmarkResult
from multitoken_estimator.training.ConceptTrainer import (
    ConceptTrainer,
    ConceptTrainerOptions,
)


@dataclass
class TrainingStrategy:
    name: str
    run_fn: Callable[[], list[Concept]]
    relation_category_map: dict[str, str]
    seed: int | float | str
    metadata: dict[str, Any]


T = TypeVar("T", bound="ConceptTrainerOptions")


def strategy_from_trainer(
    trainer: ConceptTrainer[T],
    strategy_name: str,
    opts: T,
    seed: int | float | str,
    save_progress_dir: Optional[str | Path] = None,
    force_retrain_all: bool = False,
) -> TrainingStrategy:
    """
    Create a training strategy from a trainer and a strategy name
    """
    save_progress_path = None
    if save_progress_dir is not None:
        save_progress_path = Path(save_progress_dir) / f"{strategy_name}-seed-{seed}.pt"

    def run_fn() -> list[Concept]:
        return trainer.train_all(
            opts,
            save_progress_path=save_progress_path,
            force_retrain_all=force_retrain_all,
        )

    return TrainingStrategy(
        strategy_name,
        run_fn,
        seed=seed,
        relation_category_map=trainer.relation_category_map,
        metadata=asdict(opts),
    )


def benchmark_strategies(
    strategies: list[TrainingStrategy],
    evaluator: Evaluator,
    save_progress_dir: Optional[str | Path] = None,
    force_rerun: bool = False,
    verbose: bool = True,
    seed: int | float | str = 42,
) -> dict[str, BenchmarkResult]:
    """
    Run a list of training strategies and return the results
    """
    results = {}

    if save_progress_dir is not None:
        Path(save_progress_dir).mkdir(parents=True, exist_ok=True)

    save_progress_path = _save_progress_path(
        save_progress_dir, "benchmark-results", seed=seed
    )
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
    strategy: TrainingStrategy, evaluator: Evaluator, verbose: bool = True
) -> BenchmarkResult:
    """
    Run a training strategy and return the results
    """
    train_start = time()
    concepts = strategy.run_fn()
    log_or_print(
        f"Training strategy {strategy.name} took {time() - train_start:.2f}s", verbose
    )

    eval_start = time()
    relation_accuracy = evaluator.evaluate_accuracy(concepts, verbose=verbose)
    relation_causality = evaluator.evaluate_causality(concepts, verbose=verbose)
    log_or_print(
        f"Evaluating strategy {strategy.name} took {time() - eval_start:.2f}s", verbose
    )

    results = BenchmarkResult(
        strategy.name,
        concepts=concepts,
        relation_accuracy=relation_accuracy,
        relation_causality=relation_causality,
        relation_category_map=strategy.relation_category_map,
        seed=strategy.seed,
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
    for category, accuracy in result.accuracy_by_category.items():
        print(f"\t{category}: {accuracy:.3f}")
    print("causality by LRE type")
    for category, causality in result.causality_by_category.items():
        print(f"\t{category}: {causality:.3f}")
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
    print("accuracy per relation")
    for relation, accuracy in result.accuracy_by_relation.items():
        print(f"\t{relation}: {accuracy:.3f}")
    print("causality per relation")
    for relation, causality in result.causality_by_relation.items():
        print(f"\t{relation}: {causality:.3f}")


def _save_progress_path(
    save_progress_dir: str | Path | None,
    base_name: str,
    seed: int | float | str,
    ext: str = "pt",
) -> Path | None:
    if save_progress_dir is None:
        return None
    return Path(save_progress_dir) / f"{base_name}-seed-{seed}.{ext}"
