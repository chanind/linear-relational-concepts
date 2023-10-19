from tokenizers import Tokenizer
from torch import nn

from multitoken_estimator.database import Database
from multitoken_estimator.lib.layer_matching import LayerMatcher
from multitoken_estimator.lib.logger import log_or_print
from multitoken_estimator.lib.util import sample_or_all
from multitoken_estimator.lib.verify_answers_match_expected import (
    verify_answers_match_expected,
)
from multitoken_estimator.LinearRelationalEmbedding import LinearRelationalEmbedding
from multitoken_estimator.PromptGenerator import (
    DEFAULT_ENTITY_MODIFIERS,
    PromptGenerator,
)
from multitoken_estimator.training.train_lre import ObjectAggregation, train_lre


class LreTrainer:
    model: nn.Module
    tokenizer: Tokenizer
    layer_matcher: LayerMatcher
    database: Database
    prompt_generator: PromptGenerator

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        layer_matcher: LayerMatcher,
        database: Database,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_matcher = layer_matcher
        self.database = database
        self.prompt_generator = PromptGenerator(database)

    def train_relation_lre(
        self,
        relation: str,
        subject_layer: int,
        object_layer: int,
        n_lre_training_prompts: int = 5,
        augment_lre_prompts: bool = False,
        object_aggregation: ObjectAggregation = "mean",
        n_fsl_prompts: int = 5,
        batch_size: int = 8,
        max_consider_prompts: int = 100,
        verbose: bool = True,
    ) -> LinearRelationalEmbedding:
        lre_modifiers = DEFAULT_ENTITY_MODIFIERS if augment_lre_prompts else None
        relation_prompts = list(
            self.prompt_generator.generate_prompts_for_relation(
                relation_name=relation,
                num_fsl_examples=n_fsl_prompts,
                entity_modifiers=lre_modifiers,
            )
        )
        relation_prompts = sample_or_all(relation_prompts, max_consider_prompts)
        check_prompt_results = verify_answers_match_expected(
            self.model,
            self.tokenizer,
            prompts=[prompt.text for prompt in relation_prompts],
            expected_answers=[prompt.answer for prompt in relation_prompts],
            batch_size=batch_size,
        )
        valid_prompts = [
            prompt
            for prompt, result in zip(relation_prompts, check_prompt_results)
            if result.answer_matches_expected
        ]
        if len(valid_prompts) == 0:
            raise ValueError(
                f"None of the prompts for {relation} returned the expected answer."
            )
        log_or_print(f"Training LRE for {relation}", verbose=verbose)
        training_prompts = sample_or_all(list(valid_prompts), n_lre_training_prompts)
        return train_lre(
            self.model,
            self.tokenizer,
            layer_matcher=self.layer_matcher,
            relation_name=relation,
            subject_layer=subject_layer,
            object_layer=object_layer,
            prompts=training_prompts,
            object_aggregation=object_aggregation,
            move_to_cpu=False,
        )
