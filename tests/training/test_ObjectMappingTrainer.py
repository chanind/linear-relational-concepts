from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from multitoken_estimator.data.data_loaders import load_lre_data
from multitoken_estimator.training.ObjectMappingTrainer import (
    ObjectMappingTrainer,
    _calc_reweighting,
)


def test_ObjectMappingTrainer(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    dataset = load_lre_data(
        {"city_in_country.json", "country_language.json", "pokemon_evolutions.json"}
    )
    trainer = ObjectMappingTrainer(model, tokenizer, "transformer.h.{num}", dataset)
    mapping = trainer.train(
        source_layer=5,
        target_layer=7,
        activations_dim=768,
        n_epochs=5,
        squeeze_dim=100,
    )
    assert mapping.activations_dim == 768
    assert mapping.source_layer == 5
    assert mapping.target_layer == 7
    assert mapping.mapping_in.weight.shape == (100, 768)
    assert mapping.mapping_out.weight.shape == (768, 100)


def test_calc_reweighting_does_nothing_if_classes_are_balanced() -> None:
    assert _calc_reweighting(0.5, 2) == 1.0
    assert _calc_reweighting(0.25, 4) == 1.0
    assert _calc_reweighting(0.2, 5) == 1.0
    assert _calc_reweighting(0.1, 10) == 1.0


def test_calc_reweighting_increases_weight_for_uncommon_classes() -> None:
    assert _calc_reweighting(0.25, 2) == 2.0
    assert _calc_reweighting(0.1, 5) == 2.0


def test_calc_reweighting_decreases_weight_for_common_classes() -> None:
    assert _calc_reweighting(0.8, 2) == 0.625
