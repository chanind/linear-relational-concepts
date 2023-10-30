from __future__ import annotations

from collections import defaultdict
from time import time
from typing import Literal, Optional

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, LlamaForCausalLM

from multitoken_estimator.data.data_loaders import load_lre_data
from multitoken_estimator.evaluation.Evaluator import Evaluator
from multitoken_estimator.lib.constants import DEFAULT_DEVICE
from multitoken_estimator.lib.logger import log_or_print
from multitoken_estimator.lib.PromptValidator import PromptValidator
from multitoken_estimator.training.benchmarking import (
    BenchmarkResult,
    TrainingStrategy,
    benchmark_strategies,
    strategy_from_trainer,
)
from multitoken_estimator.training.BenchmarkIterationsResult import (
    BenchmarkIterationsResult,
)
from multitoken_estimator.training.LreConceptTrainer import (
    LreConceptTrainer,
    LreConceptTrainerOptions,
)
from multitoken_estimator.training.train_lre import ObjectAggregation

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
    valid_prompts_cache_file: Optional[str] = None,
    object_aggregation: ObjectAggregation = "mean",
    min_test_prompts_per_relation: int = 1,
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

    prompt_validator = PromptValidator(model, tokenizer, valid_prompts_cache_file)

    dataset = load_lre_data()

    iteration_results: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for iteration_seed in iteration_seeds:
        start = time()
        log_or_print(f"Iteration seed: {iteration_seed}", verbose=verbose)
        raw_train_data, test_data = dataset.split(seed=iteration_seed)
        evaluator = Evaluator(
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
        )

        log_or_print("Filtering relations with few eval prompts", verbose=verbose)
        train_data = evaluator.filter_relations_with_few_eval_prompts(
            raw_train_data, min_relation_eval_prompts=min_test_prompts_per_relation
        )
        if valid_prompts_cache_file:
            prompt_validator.write_cache(valid_prompts_cache_file)

        lre_trainer = LreConceptTrainer(
            model,
            tokenizer,
            LAYER_MATCHER,
            train_data,
            prompt_validator=prompt_validator,
        )

        strategies: list[TrainingStrategy] = [
            strategy_from_trainer(
                lre_trainer,
                "var1",
                LreConceptTrainerOptions(
                    layer=19,
                    object_layer=24,
                    object_aggregation=object_aggregation,
                    verbose=verbose,
                    inv_lre_rank=INV_LRE_RANK,
                    n_fsl_prompts=4,
                ),
                save_progress_dir=save_progress_dir,
                seed=iteration_seed,
                force_retrain_all=force_rerun,
            ),
        ]

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
