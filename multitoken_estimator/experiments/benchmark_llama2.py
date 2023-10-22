from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Literal, Optional

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, LlamaForCausalLM

from multitoken_estimator.data.data_loaders import load_lre_data
from multitoken_estimator.lib.constants import DEFAULT_DEVICE
from multitoken_estimator.lib.logger import log_or_print
from multitoken_estimator.training.benchmarking import (
    BenchmarkIterationsResult,
    BenchmarkResult,
    TrainingStrategy,
    benchmark_strategies,
    strategy_from_trainer,
)
from multitoken_estimator.training.LreEvaluator import LreEvaluator
from multitoken_estimator.training.LreTrainer import LreTrainer

BATCH_SIZE = 8
LAYER_MATCHER = "model.layers.{num}"
INV_LRE_RANK = 100

Precision = Literal["fp16", "bf16", "fp32"]


def benchmark_llama2(
    model: Optional[LlamaForCausalLM] = None,
    tokenizer: Optional[Tokenizer] = None,
    iteration_seeds: list[int] = [42, 43, 44],
    save_progress_dir: Optional[str] = None,
    force_rerun: bool = False,
    device: torch.device = DEFAULT_DEVICE,
    verbose: bool = True,
    batch_size: int = BATCH_SIZE,
    causality_magnitude_multiplier: float = 0.065,
    causality_edit_single_layer_only: bool = False,
    causality_use_remove_concept_projection_magnitude: bool = False,
    precision: Precision = "fp16",
    eval_zs_prompts: bool = True,
) -> dict[str, BenchmarkIterationsResult]:
    if model is None:
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        if precision == "bf16":
            model = model.bfloat16()
        elif precision == "fp16":
            model = model.half()
        model = model.to(device)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model.eval()

    shared_args: dict[str, str | float | int] = {
        "verbose": verbose,
        "force_retrain_all": force_rerun,
    }

    dataset = load_lre_data()
    iteration_results: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for iteration_seed in iteration_seeds:
        start = time()
        log_or_print(f"Iteration seed: {iteration_seed}", verbose=verbose)
        train_data, test_data = dataset.split(seed=iteration_seed)
        lre_trainer = LreTrainer(model, tokenizer, LAYER_MATCHER, train_data)

        if save_progress_dir is not None:
            Path(save_progress_dir).mkdir(parents=True, exist_ok=True)

        def save_progress_path(strategy_name: str) -> Optional[Path]:
            if save_progress_dir:
                return (
                    Path(save_progress_dir) / f"{strategy_name}-seed{iteration_seed}.pt"
                )
            return None

        evaluator = LreEvaluator(
            model=model,
            tokenizer=tokenizer,
            layer_matcher=LAYER_MATCHER,
            dataset=test_data,
            batch_size=batch_size,
            inv_lre_rank=INV_LRE_RANK,
            use_zs_prompts=eval_zs_prompts,
            causality_magnitude_multiplier=causality_magnitude_multiplier,
            causality_edit_single_layer_only=causality_edit_single_layer_only,
            causality_use_remove_concept_projection_magnitude=causality_use_remove_concept_projection_magnitude,
        )

        strategies: list[TrainingStrategy] = [
            strategy_from_trainer(
                lre_trainer,
                f"var1-seed{iteration_seed}",
                {
                    "save_progress_path": save_progress_path("var1"),
                    "subject_layer": 19,
                    "object_layer": 24,
                    "object_aggregation": "mean",
                    **shared_args,
                },
            ),
            strategy_from_trainer(
                lre_trainer,
                f"var2-seed{iteration_seed}",
                {
                    "save_progress_path": save_progress_path("var2"),
                    "subject_layer": 19,
                    "object_layer": 24,
                    "object_aggregation": "first_token",
                    **shared_args,
                },
            ),
            strategy_from_trainer(
                lre_trainer,
                f"original-seed{iteration_seed}",
                {
                    "save_progress_path": save_progress_path("original"),
                    "subject_layer": 19,
                    "object_layer": 31,
                    "object_aggregation": "first_token",
                    **shared_args,
                },
            ),
        ]

        eval_results = benchmark_strategies(
            strategies,
            evaluator=evaluator,
            save_progress_path=save_progress_path("overall-benchmark"),
            force_rerun=force_rerun,
            verbose=verbose,
        )

        for strategy, result in eval_results.items():
            log_or_print(f"iteration took {time() - start} seconds", verbose=verbose)
            iteration_results[strip_iteration_info(strategy)].append(result)
    return {
        strategy: BenchmarkIterationsResult(iteration_results)
        for strategy, iteration_results in iteration_results.items()
    }


def strip_iteration_info(strategy_name: str) -> str:
    # remove -seed## from strategy name
    return re.sub(r"-seed\d+$", "", strategy_name)
