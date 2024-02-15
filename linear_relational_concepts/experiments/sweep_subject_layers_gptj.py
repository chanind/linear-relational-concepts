from __future__ import annotations

from typing import Literal, Optional

import torch
from linear_relational.training.train_lre import ObjectAggregation
from tokenizers import Tokenizer
from transformers import AutoTokenizer, GPTJForCausalLM

from linear_relational_concepts.data.data_loaders import load_lre_data
from linear_relational_concepts.lib.constants import DEFAULT_DEVICE
from linear_relational_concepts.training.LreConceptTrainer import (
    LreConceptTrainerOptions,
)
from linear_relational_concepts.training.sweep_lre_params import (
    SweepResult,
    sweep_lre_params,
)

BATCH_SIZE = 8
LAYER_MATCHER = "transformer.h.{num}"

Precision = Literal["fp16", "bf16", "fp32"]


def sweep_subject_layers_gptj(
    model: Optional[GPTJForCausalLM] = None,
    tokenizer: Optional[Tokenizer] = None,
    iteration_seeds: list[int | str] = [42, 43, 44, 45, 46],
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
    object_layer: int = 20,
    inv_lre_rank: int = 192,
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

    def opts_from_subject_layer(
        subject_layer: int,
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
        name="sweep_subject_layers_gptj",
        param_name="subject_layer",
        param_values=list(range(10, object_layer)),
        dataset=load_lre_data(),
        iteration_seeds=iteration_seeds,
        trainer_opts_fn=opts_from_subject_layer,
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
