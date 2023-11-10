from __future__ import annotations

from typing import Literal, Optional

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, GPTJForCausalLM

from multitoken_estimator.data.data_loaders import load_lre_data
from multitoken_estimator.lib.constants import DEFAULT_DEVICE
from multitoken_estimator.training.LreConceptTrainer import LreConceptTrainerOptions
from multitoken_estimator.training.sweep_lre_params import SweepResult, sweep_lre_params
from multitoken_estimator.training.train_lre import ObjectAggregation

BATCH_SIZE = 8
LAYER_MATCHER = "transformer.h.{num}"

Precision = Literal["fp16", "bf16", "fp32"]


def sweep_ranks_gptj(
    model: Optional[GPTJForCausalLM] = None,
    tokenizer: Optional[Tokenizer] = None,
    iteration_seeds: list[int | str] = [42, 43, 44, 44, 45],
    save_progress_dir: Optional[str] = None,
    force_rerun: bool = False,
    device: torch.device = DEFAULT_DEVICE,
    verbose: bool = True,
    batch_size: int = BATCH_SIZE,
    causality_magnitude_multiplier: float = 0.05,
    causality_edit_single_layer_only: bool = False,
    causality_use_remove_concept_projection_magnitude: bool = False,
    precision: Precision = "fp16",
    eval_zs_prompts: bool = True,
    valid_prompts_cache_file: Optional[str] = None,
    min_test_prompts_per_relation: int = 1,
    subject_layer: int = 15,
    object_layer: int = 19,
    object_aggregation: ObjectAggregation = "mean",
) -> SweepResult[int]:
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

    def opts_from_inv_lre_rank(
        inv_lre_rank: int,
        iteration_seed: int | str,
    ) -> LreConceptTrainerOptions:
        return LreConceptTrainerOptions(
            layer=subject_layer,
            object_layer=object_layer,
            object_aggregation=object_aggregation,
            sampling_method="random",
            inv_lre_rank=inv_lre_rank,
            seed=iteration_seed,
        )

    return sweep_lre_params(
        model,
        tokenizer,
        LAYER_MATCHER,
        name="sweep_inv_lre_ranks_gptj",
        param_name="inv_lre_rank",
        param_values=[4, 8, 16, 32, 64, 128, 160, 192, 224, 256, 512, 1024, 2048],
        dataset=load_lre_data(),
        iteration_seeds=iteration_seeds,
        trainer_opts_fn=opts_from_inv_lre_rank,
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
