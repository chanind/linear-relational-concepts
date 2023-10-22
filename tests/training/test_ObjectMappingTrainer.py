from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from multitoken_estimator.data.data_loaders import load_lre_data
from multitoken_estimator.training.ObjectMappingTrainer import ObjectMappingTrainer


def test_ObjectMappingTrainer(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    dataset = load_lre_data(
        {"city_in_country.json", "fruit_inside_color.json", "pokemon_evolution.json"}
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
