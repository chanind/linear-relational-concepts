from unittest.mock import Mock

import pytest
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from linear_relational_concepts.data.RelationDataset import (
    Relation,
    RelationDataset,
    Sample,
)
from linear_relational_concepts.evaluation.Evaluator import Evaluator
from linear_relational_concepts.training.LreConceptTrainer import (
    InvLreTrainingRunManager,
    LreConceptTrainer,
    LreConceptTrainerOptions,
    _num_balanced_sample_targets,
    _sample_balanced,
)
from tests.helpers import create_prompt


def test_LreConceptTrainer_train_all(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    dataset = RelationDataset()
    dataset.add_relation(
        Relation(
            name="located_in_country", templates=("{} is located in the country of",)
        )
    )
    japan_cities = [
        "Tokyo",
        "Osaka",
        "Nagoya",
        "Hiroshima",
        "Yokohama",
        "Kyoto",
        "Nagasaki",
        "Kobe",
        "Kitashima",
        "Kyushu",
    ]
    china_cities = [
        "Beijing",
        "Shanghai",
        "Nanjing",
        "Hangzhou",
        "Peking",
        "Qingdao",
        "Chongqing",
        "Changsha",
        "Wuhan",
        "Chengdu",
    ]
    for city in japan_cities:
        dataset.add_sample(
            Sample(relation="located_in_country", subject=city, object="Japan")
        )
    for city in china_cities:
        dataset.add_sample(
            Sample(relation="located_in_country", subject=city, object="China")
        )

    trainer = LreConceptTrainer(
        model, tokenizer, layer_matcher="transformer.h.{num}", dataset=dataset
    )
    opts = LreConceptTrainerOptions(layer=8, object_layer=10)

    concepts = trainer.train_all(opts)

    assert len(concepts) == 2
    for concept in concepts:
        assert concept.layer == 8
        assert concept.vector.shape == (768,)
        assert concept.vector.norm() == pytest.approx(1.0)

    # evaluating on the train set should score highly
    evaluator = Evaluator(
        model,
        tokenizer,
        layer_matcher="transformer.h.{num}",
        dataset=dataset,
        prompt_validator=trainer.prompt_validator,
    )
    accuracy_results = evaluator.evaluate_accuracy(concepts, verbose=False)
    assert accuracy_results["located_in_country"].accuracy > 0.9


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
        hidden_layer_matcher="transformer.h.{num}",
        relation_name="located_in_country",
        subject_layer=5,
        object_layer=8,
        prompts_by_object=prompts_by_object,
        object_aggregation="mean",
        inv_lre_rank=10,
        max_train_samples=1,
        sampling_method="balanced_ceil",
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
        hidden_layer_matcher="transformer.h.{num}",
        relation_name="located_in_country",
        subject_layer=5,
        object_layer=8,
        prompts_by_object=prompts_by_object,
        object_aggregation="mean",
        inv_lre_rank=10,
        max_train_samples=2,
        sampling_method="balanced_ceil",
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


def test_InvLreTrainingRunManager_get_inv_lre_with_negative_object_layer(
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
        hidden_layer_matcher="transformer.h.{num}",
        relation_name="located_in_country",
        subject_layer=5,
        object_layer=-1,
        prompts_by_object=prompts_by_object,
        object_aggregation="mean",
        inv_lre_rank=10,
        max_train_samples=2,
        sampling_method="balanced_ceil",
        seed=42,
    )

    inv_lre = inv_lre_manager.get_inv_lre_for_object("Japan")
    assert inv_lre.relation == "located_in_country"
    assert inv_lre.rank == 10
    assert inv_lre.subject_layer == 5
    assert inv_lre.object_layer == 11
    assert inv_lre.object_aggregation == "mean"


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
        hidden_layer_matcher="transformer.h.{num}",
        relation_name="located_in_country",
        subject_layer=5,
        object_layer=8,
        prompts_by_object=prompts_by_object,
        object_aggregation="mean",
        inv_lre_rank=10,
        max_train_samples=3,
        sampling_method="balanced_floor",
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


def test_num_balanced_sample_targets() -> None:
    prompts_by_object = {
        "Japan": [
            create_prompt("Tokyo", "Japan"),
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
    assert _num_balanced_sample_targets(
        prompts_by_object, 3, floor_num_obj_samples=True
    ) == (1, 2)
    assert _num_balanced_sample_targets(
        prompts_by_object, 3, floor_num_obj_samples=False
    ) == (1, 2)
    assert _num_balanced_sample_targets(
        prompts_by_object, 10, floor_num_obj_samples=True
    ) == (3, 7)
    assert _num_balanced_sample_targets(
        prompts_by_object, 10, floor_num_obj_samples=False
    ) == (4, 6)


def test_samples_balanced() -> None:
    prompts_by_object = {
        "Japan": [
            create_prompt("Tokyo", "Japan"),
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
    samples = _sample_balanced(
        "Japan", prompts_by_object, 3, floor_num_obj_samples=True, seed=42
    )
    assert len(samples) == 3
    assert len([s for s in samples if s.object_name == "Japan"]) == 1


def test_samples_balanced_with_too_few_cur_obj_samples() -> None:
    prompts_by_object = {
        "Japan": [
            create_prompt("Tokyo", "Japan"),
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
    samples = _sample_balanced(
        "Japan", prompts_by_object, 5, floor_num_obj_samples=False, seed=42
    )
    # should be aiming for 2 + 3, but only can do 1 + 3
    assert len(samples) == 4
    assert len([s for s in samples if s.object_name == "Japan"]) == 1


def test_samples_balanced_with_too_few_samples() -> None:
    prompts_by_object = {
        "Japan": [
            create_prompt("Tokyo", "Japan"),
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
    samples = _sample_balanced(
        "Japan", prompts_by_object, 20, floor_num_obj_samples=False, seed=42
    )
    assert len(samples) == 5
    assert len([s for s in samples if s.object_name == "Japan"]) == 1
