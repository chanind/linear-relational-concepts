from dataclasses import dataclass, replace
from typing import Callable, Generic, Optional, Sequence, TypeVar

import torch
from tokenizers import Tokenizer

from multitoken_estimator.data.RelationDataset import RelationDataset
from multitoken_estimator.lib.layer_matching import LayerMatcher
from multitoken_estimator.lib.PromptValidator import PromptValidator
from multitoken_estimator.training.benchmarking import (
    TrainingStrategy,
    benchmark_iterations,
    strategy_from_trainer,
)
from multitoken_estimator.training.BenchmarkIterationsResult import (
    BenchmarkIterationsResult,
)
from multitoken_estimator.training.LreConceptTrainer import (
    LreConceptTrainer,
    LreConceptTrainerOptions,
)

PVal = TypeVar("PVal", int, float, str)


@dataclass
class SweepResult(Generic[PVal]):
    name: str
    param_name: str
    param_values: list[PVal]
    results: list[BenchmarkIterationsResult]


def sweep_lre_params(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    layer_matcher: LayerMatcher,
    name: str,
    param_name: str,
    param_values: Sequence[PVal],
    dataset: RelationDataset,
    # fn which takes the param value and the seed and returns the trainer options
    trainer_opts_fn: Callable[
        [PVal, int | str],
        LreConceptTrainerOptions,
    ],
    iteration_seeds: list[int | str] = [42, 43, 44, 44, 45],
    verbose: bool = True,
    batch_size: int = 8,
    save_progress_dir: Optional[str] = None,
    force_rerun: bool = False,
    causality_magnitude_multiplier: float = 0.09,
    causality_edit_single_layer_only: bool = False,
    causality_use_remove_concept_projection_magnitude: bool = False,
    valid_prompts_cache_file: Optional[str] = None,
    eval_zs_prompts: bool = True,
    dataset_train_ration: float = 0.5,
    min_test_prompts_per_relation: int = 1,
) -> SweepResult[PVal]:
    param_to_strategy_name: dict[PVal, str] = {}

    def build_strategies(
        prompt_validator: PromptValidator, dataset: RelationDataset, seed: int | str
    ) -> list[TrainingStrategy]:
        trainer = LreConceptTrainer(
            model=model,
            tokenizer=tokenizer,
            layer_matcher=layer_matcher,
            dataset=dataset,
            prompt_validator=prompt_validator,
        )
        strategies: list[TrainingStrategy] = []
        for param_value in param_values:
            strategy_name = f"{param_name}-{param_value}"
            param_to_strategy_name[param_value] = strategy_name
            trainer_opts = trainer_opts_fn(param_value, seed)
            strategy = strategy_from_trainer(
                trainer,
                strategy_name,
                trainer_opts,
                seed=seed,
                save_progress_dir=save_progress_dir,
                force_retrain_all=force_rerun,
            )
            strategies.append(replace(strategy, name=strategy_name))
        return strategies

    iter_results = benchmark_iterations(
        model,
        tokenizer,
        layer_matcher,
        dataset=dataset,
        iteration_seeds=iteration_seeds,
        strategies_fn=build_strategies,
        verbose=verbose,
        batch_size=batch_size,
        save_progress_dir=save_progress_dir,
        force_rerun=force_rerun,
        causality_magnitude_multiplier=causality_magnitude_multiplier,
        causality_edit_single_layer_only=causality_edit_single_layer_only,
        causality_use_remove_concept_projection_magnitude=causality_use_remove_concept_projection_magnitude,
        valid_prompts_cache_file=valid_prompts_cache_file,
        eval_zs_prompts=eval_zs_prompts,
        dataset_train_ration=dataset_train_ration,
        min_test_prompts_per_relation=min_test_prompts_per_relation,
    )

    return SweepResult(
        name=name,
        param_name=param_name,
        param_values=list(param_values),
        results=[iter_results[param_to_strategy_name[pval]] for pval in param_values],
    )
