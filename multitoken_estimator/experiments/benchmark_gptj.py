from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Literal, Optional

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, GPTJForCausalLM

from multitoken_estimator.data.data_loaders import load_lre_data
from multitoken_estimator.lib.constants import DEFAULT_DEVICE
from multitoken_estimator.lib.logger import log_or_print
from multitoken_estimator.lib.PromptValidator import PromptValidator
from multitoken_estimator.PromptGenerator import PromptGenerator
from multitoken_estimator.training.benchmarking import (
    BenchmarkIterationsResult,
    BenchmarkResult,
    TrainingStrategy,
    benchmark_strategies,
    strategy_from_trainer,
)
from multitoken_estimator.training.LreEvaluator import LreEvaluator
from multitoken_estimator.training.RelationalConceptEstimatorTrainer import (
    RelationalConceptEstimatorTrainer,
)
from multitoken_estimator.training.train_lre import ObjectAggregation

BATCH_SIZE = 8
LAYER_MATCHER = "transformer.h.{num}"
INV_LRE_RANK = 100
ACTIVATIONS_DIM = 4096

Precision = Literal["fp16", "bf16", "fp32"]


def benchmark_gptj(
    model: Optional[GPTJForCausalLM] = None,
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
    valid_prompts_cache_file: Optional[str] = None,
    prefilter_objects: bool = True,
    object_aggregation: ObjectAggregation = "mean",
) -> dict[str, BenchmarkIterationsResult]:
    if model is None:
        model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        if precision == "bf16":
            model = model.bfloat16()
        elif precision == "fp16":
            model = model.half()
        model = model.to(device)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    model.eval()

    prompt_validator = PromptValidator(model, tokenizer, valid_prompts_cache_file)

    shared_args: dict[str, str | float | int] = {
        "verbose": verbose,
        "force_retrain_all": force_rerun,
        "activations_dim": ACTIVATIONS_DIM,
        "inv_lre_rank": INV_LRE_RANK,
        "mapping_source_layer": 15,
        "object_aggregation": object_aggregation,
    }

    dataset = load_lre_data()
    valid_objects_by_relation: dict[str, set[str]] | None = None
    if prefilter_objects:
        prompt_generator = PromptGenerator(dataset)
        all_prompts = prompt_generator.generate_prompts_for_all_relations(
            num_fsl_examples=5,
            entity_modifiers=None,
        )
        valid_prompts = prompt_validator.filter_prompts(
            all_prompts, batch_size=batch_size
        )
        valid_objects_by_relation = defaultdict(set)
        for prompt in valid_prompts:
            valid_objects_by_relation[prompt.relation_name].add(prompt.object_name)

    iteration_results: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for iteration_seed in iteration_seeds:
        start = time()
        log_or_print(f"Iteration seed: {iteration_seed}", verbose=verbose)
        train_data, test_data = dataset.split(seed=iteration_seed)
        lre_trainer = RelationalConceptEstimatorTrainer(
            model,
            tokenizer,
            LAYER_MATCHER,
            train_data,
            prompt_validator=prompt_validator,
        )

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
            use_zs_prompts=eval_zs_prompts,
            causality_magnitude_multiplier=causality_magnitude_multiplier,
            causality_edit_single_layer_only=causality_edit_single_layer_only,
            causality_use_remove_concept_projection_magnitude=causality_use_remove_concept_projection_magnitude,
            prompt_validator=prompt_validator,
            valid_objects_by_relation=valid_objects_by_relation,
        )

        strategies: list[TrainingStrategy] = [
            strategy_from_trainer(
                lre_trainer,
                f"var1-seed{iteration_seed}",
                {
                    "save_progress_path": save_progress_path("var1"),
                    "subject_layer": 15,
                    "object_layer": 22,
                    "object_aggregation": "mean",
                    **shared_args,
                },
            ),
            strategy_from_trainer(
                lre_trainer,
                f"var2-seed{iteration_seed}",
                {
                    "save_progress_path": save_progress_path("var2"),
                    "subject_layer": 15,
                    "object_layer": 22,
                    "object_aggregation": "first_token",
                    **shared_args,
                },
            ),
            strategy_from_trainer(
                lre_trainer,
                f"original-seed{iteration_seed}",
                {
                    "save_progress_path": save_progress_path("original"),
                    "subject_layer": 15,
                    "object_layer": 27,
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
        if valid_prompts_cache_file:
            prompt_validator.write_cache(valid_prompts_cache_file)

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
