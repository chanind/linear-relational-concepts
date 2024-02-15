from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time
from typing import Any, Callable, Optional, TypeVar

import torch
from linear_relational import Concept
from tokenizers import Tokenizer

from linear_relational_concepts.data.RelationDataset import RelationDataset
from linear_relational_concepts.evaluation.Evaluator import Evaluator
from linear_relational_concepts.lib.layer_matching import LayerMatcher
from linear_relational_concepts.lib.logger import log_or_print
from linear_relational_concepts.lib.PromptValidator import PromptValidator
from linear_relational_concepts.training.BenchmarkIterationsResult import (
    BenchmarkIterationsResult,
)
from linear_relational_concepts.training.BenchmarkResult import (
    BenchmarkResult,
    print_benchmark_result,
)
from linear_relational_concepts.training.ConceptTrainer import (
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


def benchmark_iterations(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    layer_matcher: LayerMatcher,
    dataset: RelationDataset,
    iteration_seeds: list[int | str],
    # fn which takes a prompt validator, a dataset, a seed
    strategies_fn: Callable[
        [PromptValidator, RelationDataset, int | str],
        list[TrainingStrategy],
    ],
    verbose: bool = True,
    batch_size: int = 8,
    save_progress_dir: Optional[str] = None,
    force_rerun: bool = False,
    causality_magnitude_multiplier: float = 0.05,
    causality_edit_single_layer_only: bool = False,
    causality_use_remove_concept_projection_magnitude: bool = False,
    valid_prompts_cache_file: Optional[str] = None,
    eval_zs_prompts: bool = True,
    dataset_train_ration: float = 0.5,
    min_test_prompts_per_relation: int = 1,
) -> dict[str, BenchmarkIterationsResult]:
    prompt_validator = PromptValidator(model, tokenizer, valid_prompts_cache_file)

    iteration_results: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for iteration_seed in iteration_seeds:
        start = time()
        log_or_print(f"Iteration seed: {iteration_seed}", verbose=verbose)
        raw_train_data, test_data = dataset.split(
            seed=iteration_seed, train_portion=dataset_train_ration
        )
        evaluator = Evaluator(
            model=model,
            tokenizer=tokenizer,
            layer_matcher=layer_matcher,
            dataset=test_data,
            batch_size=batch_size,
            use_zs_prompts=eval_zs_prompts,
            causality_magnitude_multiplier=causality_magnitude_multiplier,
            causality_edit_single_layer_only=causality_edit_single_layer_only,
            causality_use_remove_concept_projection_magnitude=causality_use_remove_concept_projection_magnitude,
            prompt_validator=prompt_validator,
        )

        log_or_print("Filtering relations with few eval prompts", verbose=verbose)
        train_data = evaluator.filter_relations_with_few_eval_prompts(
            raw_train_data, min_relation_eval_prompts=min_test_prompts_per_relation
        )
        if valid_prompts_cache_file:
            prompt_validator.write_cache(valid_prompts_cache_file)

        strategies = strategies_fn(prompt_validator, train_data, iteration_seed)

        results = benchmark_strategies(
            strategies,
            evaluator=evaluator,
            save_progress_dir=save_progress_dir,
            force_rerun=force_rerun,
            verbose=verbose,
            seed=iteration_seed,
        )
        if valid_prompts_cache_file:
            prompt_validator.write_cache(valid_prompts_cache_file)

        for strategy, result in results.items():
            log_or_print(f"iteration took {time() - start} seconds", verbose=verbose)
            iteration_results[strategy].append(result)
    return {
        strategy: BenchmarkIterationsResult(iteration_results)
        for strategy, iteration_results in iteration_results.items()
    }


def _save_progress_path(
    save_progress_dir: str | Path | None,
    base_name: str,
    seed: int | float | str,
    ext: str = "pt",
) -> Path | None:
    if save_progress_dir is None:
        return None
    return Path(save_progress_dir) / f"{base_name}-seed-{seed}.{ext}"
