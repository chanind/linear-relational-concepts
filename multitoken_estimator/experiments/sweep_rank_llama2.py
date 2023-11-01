from __future__ import annotations

from typing import Literal, Optional

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, LlamaForCausalLM

from multitoken_estimator.data.data_loaders import load_lre_data
from multitoken_estimator.data.RelationDataset import RelationDataset
from multitoken_estimator.lib.constants import DEFAULT_DEVICE
from multitoken_estimator.lib.layer_matching import collect_matching_layers
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
from multitoken_estimator.training.train_lre import ObjectAggregation

BATCH_SIZE = 8
LAYER_MATCHER = "model.layers.{num}"

Precision = Literal["fp16", "bf16", "fp32"]


def sweep_rank_llama2(
    model: Optional[LlamaForCausalLM] = None,
    tokenizer: Optional[Tokenizer] = None,
    iteration_seeds: list[int | str] = [42, 43, 44, 44, 45],
    save_progress_dir: Optional[str] = None,
    force_rerun: bool = False,
    device: torch.device = DEFAULT_DEVICE,
    verbose: bool = True,
    batch_size: int = BATCH_SIZE,
    causality_magnitude_multiplier: float = 0.09,
    causality_edit_single_layer_only: bool = False,
    causality_use_remove_concept_projection_magnitude: bool = False,
    precision: Precision = "fp16",
    eval_zs_prompts: bool = True,
    valid_prompts_cache_file: Optional[str] = None,
    min_test_prompts_per_relation: int = 1,
    subject_layer: int = 19,
    object_layer: int = 24,
    object_aggregation: ObjectAggregation = "first_token",
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

    def get_strategies(
        prompt_validator: PromptValidator,
        train_dataset: RelationDataset,
        iteration_seed: int | str,
    ) -> list[TrainingStrategy]:
        lre_trainer = LreConceptTrainer(
            model,
            tokenizer,
            LAYER_MATCHER,
            train_dataset,
            prompt_validator=prompt_validator,
        )

        strategies = [
            strategy_from_trainer(
                lre_trainer,
                f"lre-{object_aggregation}-{subject_layer}-{inv_lre_rank}-{object_layer}",
                LreConceptTrainerOptions(
                    layer=subject_layer,
                    object_layer=object_layer,
                    object_aggregation=object_aggregation,
                    sampling_method="random",
                    inv_lre_rank=inv_lre_rank,
                ),
                save_progress_dir=save_progress_dir,
                seed=iteration_seed,
                force_retrain_all=force_rerun,
            )
            for inv_lre_rank in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        ]
        return strategies

    return benchmark_iterations(
        model,
        tokenizer,
        LAYER_MATCHER,
        dataset=load_lre_data(),
        iteration_seeds=iteration_seeds,
        strategies_fn=get_strategies,
        batch_size=batch_size,
        eval_zs_prompts=eval_zs_prompts,
        min_test_prompts_per_relation=min_test_prompts_per_relation,
        causality_magnitude_multiplier=causality_magnitude_multiplier,
        causality_edit_single_layer_only=causality_edit_single_layer_only,
        causality_use_remove_concept_projection_magnitude=causality_use_remove_concept_projection_magnitude,
        valid_prompts_cache_file=valid_prompts_cache_file,
        verbose=verbose,
        save_progress_dir=save_progress_dir,
        force_rerun=force_rerun,
    )
