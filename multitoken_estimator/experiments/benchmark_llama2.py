from __future__ import annotations

from typing import Literal, Optional

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, LlamaForCausalLM

from multitoken_estimator.data.data_loaders import load_lre_data
from multitoken_estimator.data.RelationDataset import RelationDataset
from multitoken_estimator.lib.constants import DEFAULT_DEVICE
from multitoken_estimator.lib.PromptValidator import PromptValidator
from multitoken_estimator.training.AvgConceptTrainer import (
    AvgConceptTrainer,
    AvgConceptTrainerOptions,
)
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
from multitoken_estimator.training.SvmConceptTrainer import (
    SvmConceptTrainer,
    SvmConceptTrainerOptions,
)

BATCH_SIZE = 8
LAYER_MATCHER = "model.layers.{num}"

Precision = Literal["fp16", "bf16", "fp32"]


def benchmark_llama2(
    model: Optional[LlamaForCausalLM] = None,
    tokenizer: Optional[Tokenizer] = None,
    iteration_seeds: list[int | str] = [42, 43, 44, 45, 46],
    save_progress_dir: Optional[str] = None,
    force_rerun_all: bool = False,
    rerun_evaluation: bool = False,
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
    skip_svm: bool = False,
    skip_avg: bool = False,
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
        svm_trainer = SvmConceptTrainer(
            model,
            tokenizer,
            LAYER_MATCHER,
            train_dataset,
            prompt_validator=prompt_validator,
        )
        avg_trainer = AvgConceptTrainer(
            model,
            tokenizer,
            LAYER_MATCHER,
            train_dataset,
            prompt_validator=prompt_validator,
        )

        strategies: list[TrainingStrategy] = [
            strategy_from_trainer(
                lre_trainer,
                "lre-og-rand-r192-s17-20s",
                LreConceptTrainerOptions(
                    layer=17,
                    object_layer=-1,
                    object_aggregation="first_token",
                    sampling_method="random",
                    max_lre_training_samples=20,
                    inv_lre_rank=192,
                ),
                save_progress_dir=save_progress_dir,
                seed=iteration_seed,
                force_retrain_all=force_rerun_all,
            ),
            strategy_from_trainer(
                lre_trainer,
                "lre-ft-rand-r192-s17-o21-20s",
                LreConceptTrainerOptions(
                    layer=17,
                    object_layer=21,
                    object_aggregation="first_token",
                    sampling_method="random",
                    max_lre_training_samples=20,
                    inv_lre_rank=192,
                ),
                save_progress_dir=save_progress_dir,
                seed=iteration_seed,
                force_retrain_all=force_rerun_all,
            ),
            strategy_from_trainer(
                lre_trainer,
                "lre-mn-rand-r192-s17-o21-20s",
                LreConceptTrainerOptions(
                    layer=17,
                    object_layer=21,
                    object_aggregation="mean",
                    sampling_method="random",
                    max_lre_training_samples=20,
                    inv_lre_rank=192,
                ),
                save_progress_dir=save_progress_dir,
                seed=iteration_seed,
                force_retrain_all=force_rerun_all,
            ),
            strategy_from_trainer(
                lre_trainer,
                "lre-ft-rand-r192-s17-o24-20s",
                LreConceptTrainerOptions(
                    layer=17,
                    object_layer=24,
                    object_aggregation="first_token",
                    sampling_method="random",
                    max_lre_training_samples=20,
                    inv_lre_rank=192,
                ),
                save_progress_dir=save_progress_dir,
                seed=iteration_seed,
                force_retrain_all=force_rerun_all,
            ),
            strategy_from_trainer(
                lre_trainer,
                "lre-mn-rand-r192-s17-o24-20s",
                LreConceptTrainerOptions(
                    layer=17,
                    object_layer=24,
                    object_aggregation="mean",
                    sampling_method="random",
                    max_lre_training_samples=20,
                    inv_lre_rank=192,
                ),
                save_progress_dir=save_progress_dir,
                seed=iteration_seed,
                force_retrain_all=force_rerun_all,
            ),
            strategy_from_trainer(
                lre_trainer,
                "lre-ft-rand-r192-s17-o22-20s",
                LreConceptTrainerOptions(
                    layer=17,
                    object_layer=22,
                    object_aggregation="first_token",
                    sampling_method="random",
                    max_lre_training_samples=20,
                    inv_lre_rank=192,
                ),
                save_progress_dir=save_progress_dir,
                seed=iteration_seed,
                force_retrain_all=force_rerun_all,
            ),
            strategy_from_trainer(
                lre_trainer,
                "lre-mn-rand-r192-s17-o22-20s",
                LreConceptTrainerOptions(
                    layer=17,
                    object_layer=22,
                    object_aggregation="mean",
                    sampling_method="random",
                    max_lre_training_samples=20,
                    inv_lre_rank=192,
                ),
                save_progress_dir=save_progress_dir,
                seed=iteration_seed,
                force_retrain_all=force_rerun_all,
            ),
        ]
        if not skip_svm:
            strategies.append(
                strategy_from_trainer(
                    svm_trainer,
                    "svm-l17",
                    SvmConceptTrainerOptions(layer=17),
                    save_progress_dir=save_progress_dir,
                    seed=iteration_seed,
                    force_retrain_all=force_rerun_all,
                )
            )
        if not skip_avg:
            strategies.append(
                strategy_from_trainer(
                    avg_trainer,
                    "avg-l17",
                    AvgConceptTrainerOptions(layer=17),
                    save_progress_dir=save_progress_dir,
                    seed=iteration_seed,
                    force_retrain_all=force_rerun_all,
                ),
            )
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
        force_rerun_all=force_rerun_all or rerun_evaluation,
    )
