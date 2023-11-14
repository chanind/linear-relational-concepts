import pytest
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from linear_relational_concepts.data.RelationDataset import (
    Relation,
    RelationDataset,
    Sample,
)
from linear_relational_concepts.evaluation.Evaluator import Evaluator
from linear_relational_concepts.training.SvmConceptTrainer import (
    SvmConceptTrainer,
    SvmConceptTrainerOptions,
)


def test_SvmConceptTrainer_train_all(
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

    trainer = SvmConceptTrainer(
        model, tokenizer, layer_matcher="transformer.h.{num}", dataset=dataset
    )
    opts = SvmConceptTrainerOptions(
        layer=10,
    )

    concepts = trainer.train_all(opts)

    assert len(concepts) == 2
    for concept in concepts:
        assert concept.layer == 10
        assert concept.vector.shape == (768,)
        assert concept.vector.norm() == pytest.approx(1.0)

    # evaluating on the train set should get perfect accuracy results
    evaluator = Evaluator(
        model,
        tokenizer,
        layer_matcher="transformer.h.{num}",
        dataset=dataset,
        prompt_validator=trainer.prompt_validator,
    )
    accuracy_results = evaluator.evaluate_accuracy(concepts, verbose=False)
    assert accuracy_results["located_in_country"].accuracy == 1.0
