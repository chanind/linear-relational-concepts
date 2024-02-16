from __future__ import annotations

from typing import Literal, Optional

import torch
from linear_relational import PromptValidator
from tokenizers import Tokenizer
from transformers import AutoTokenizer, LlamaForCausalLM

from linear_relational_concepts.data.data_loaders import load_lre_data
from linear_relational_concepts.data.RelationDataset import RelationDataset
from linear_relational_concepts.lib.constants import DEFAULT_DEVICE
from linear_relational_concepts.training.benchmarking import (
    TrainingStrategy,
    benchmark_iterations,
    strategy_from_trainer,
)
from linear_relational_concepts.training.BenchmarkIterationsResult import (
    BenchmarkIterationsResult,
)
from linear_relational_concepts.training.LreConceptTrainer import (
    LreConceptTrainer,
    LreConceptTrainerOptions,
)

BATCH_SIZE = 8
LAYER_MATCHER = "model.layers.{num}"

Precision = Literal["fp16", "bf16", "fp32"]


def benchmark_single_sample_llama2(
    model: Optional[LlamaForCausalLM] = None,
    tokenizer: Optional[Tokenizer] = None,
    iteration_seeds: list[int | str] = [42, 43, 44, 45, 46],
    save_progress_dir: Optional[str] = None,
    force_rerun: bool = False,
    device: torch.device = DEFAULT_DEVICE,
    verbose: bool = True,
    batch_size: int = BATCH_SIZE,
    causality_magnitude_multiplier: float = 0.075,
    causality_edit_single_layer_only: bool = False,
    causality_use_remove_concept_projection_magnitude: bool = False,
    precision: Precision = "fp16",
    eval_zs_prompts: bool = True,
    valid_prompts_cache_file: Optional[str] = None,
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
        strategies: list[TrainingStrategy] = [
            strategy_from_trainer(
                lre_trainer,
                "lre-oco-r192-s17-o21-1s",
                LreConceptTrainerOptions(
                    layer=17,
                    object_layer=21,
                    object_aggregation="first_token",
                    sampling_method="only_current_object",
                    max_lre_training_samples=1,
                    inv_lre_rank=192,
                ),
                save_progress_dir=save_progress_dir,
                seed=iteration_seed,
                force_retrain_all=force_rerun,
            ),
            strategy_from_trainer(
                lre_trainer,
                "lre-eco-r192-s17-o21-1s",
                LreConceptTrainerOptions(
                    layer=17,
                    object_layer=21,
                    object_aggregation="first_token",
                    sampling_method="exclude_current_object",
                    max_lre_training_samples=1,
                    inv_lre_rank=192,
                ),
                save_progress_dir=save_progress_dir,
                seed=iteration_seed,
                force_retrain_all=force_rerun,
            ),
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
