from unittest.mock import Mock

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from multitoken_estimator.training.LreConceptTrainer import InvLreTrainingRunManager
from tests.helpers import create_prompt


def test_InvLreTrainingRunManager_uses_cached_inv_lre_if_possible(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    prompts_by_object = {
        "Japan": [
            create_prompt("Tokyo", "Japan"),
            create_prompt("Osaka", "Japan"),
        ],
        "China": [
            create_prompt("Beijing", "China"),
            create_prompt("Shanghai", "China"),
        ],
    }
    precomputed_lre = Mock()
    inv_lre_manager = InvLreTrainingRunManager(
        model=model,
        tokenizer=tokenizer,
        hidden_layers_matcher="transformer.h.{num}",
        relation_name="located_in_country",
        subject_layer=5,
        object_layer=8,
        prompts_by_object=prompts_by_object,
        object_aggregation="mean",
        inv_lre_rank=10,
        n_lre_object_train_samples=0,
        n_lre_non_object_train_samples=1,
        seed=42,
    )
    inv_lre_manager._precomputed_inv_lres_by_object["Japan"] = precomputed_lre

    assert inv_lre_manager.get_inv_lre_for_object("Japan") == precomputed_lre
    assert len(inv_lre_manager._precomputed_inv_lres_by_object) == 0
    assert inv_lre_manager._remaining_objects == {"China"}


def test_InvLreTrainingRunManager_get_inv_lre_for_object_updates_precomputed_list(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    prompts_by_object = {
        "Japan": [
            create_prompt("Tokyo", "Japan"),
            create_prompt("Osaka", "Japan"),
        ],
        "China": [
            create_prompt("Beijing", "China"),
            create_prompt("Shanghai", "China"),
        ],
        "Korea": [
            create_prompt("Seoul", "Korea"),
            create_prompt("Busan", "Korea"),
        ],
    }
    inv_lre_manager = InvLreTrainingRunManager(
        model=model,
        tokenizer=tokenizer,
        hidden_layers_matcher="transformer.h.{num}",
        relation_name="located_in_country",
        subject_layer=5,
        object_layer=8,
        prompts_by_object=prompts_by_object,
        object_aggregation="mean",
        inv_lre_rank=10,
        n_lre_object_train_samples=0,
        n_lre_non_object_train_samples=1,
        seed=42,
    )

    inv_lre = inv_lre_manager.get_inv_lre_for_object("Japan")
    assert inv_lre.relation == "located_in_country"
    assert inv_lre.rank == 10
    assert inv_lre.subject_layer == 5
    assert inv_lre.object_layer == 8
    assert inv_lre.object_aggregation == "mean"

    # this should be guaranteed to match either China or Korea, since we're only
    # sampling 1 non-object sample
    assert len(inv_lre_manager._precomputed_inv_lres_by_object) == 1
    assert list(inv_lre_manager._precomputed_inv_lres_by_object.keys())[0] in [
        "China",
        "Korea",
    ]
    assert inv_lre in inv_lre_manager._precomputed_inv_lres_by_object.values()


def test_InvLreTrainingRunManager_samples_satisfy_object_constraints(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    prompts_by_object = {
        "Japan": [
            create_prompt("Tokyo", "Japan"),
            create_prompt("Osaka", "Japan"),
        ],
        "China": [
            create_prompt("Beijing", "China"),
            create_prompt("Shanghai", "China"),
        ],
        "Korea": [
            create_prompt("Seoul", "Korea"),
            create_prompt("Busan", "Korea"),
        ],
    }
    inv_lre_manager = InvLreTrainingRunManager(
        model=model,
        tokenizer=tokenizer,
        hidden_layers_matcher="transformer.h.{num}",
        relation_name="located_in_country",
        subject_layer=5,
        object_layer=8,
        prompts_by_object=prompts_by_object,
        object_aggregation="mean",
        inv_lre_rank=10,
        n_lre_object_train_samples=1,
        n_lre_non_object_train_samples=2,
        seed=42,
    )

    # the goal is 1 object sample and 2 non-object samples
    assert inv_lre_manager._samples_satisfy_object_constraints(
        "Japan",
        [
            prompts_by_object["Japan"][0],
            prompts_by_object["China"][0],
            prompts_by_object["Korea"][0],
        ],
    )

    # here there's 2 object sample and 1 non-object samples, so it should fail
    assert not inv_lre_manager._samples_satisfy_object_constraints(
        "Japan",
        [
            prompts_by_object["Japan"][0],
            prompts_by_object["Japan"][0],
            prompts_by_object["Korea"][0],
        ],
    )
