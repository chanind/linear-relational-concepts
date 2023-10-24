import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from multitoken_estimator.data.data_loaders import load_lre_data
from multitoken_estimator.LinearRelationalEmbedding import (
    InvertedLinearRelationalEmbedding,
)
from multitoken_estimator.training.ObjectMappingTrainer import ObjectMappingTrainer


def test_ObjectMappingTrainer(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    dataset = load_lre_data({"city_in_country.json"})
    trainer = ObjectMappingTrainer(model, tokenizer, "transformer.h.{num}", dataset)
    mapping = trainer.train(
        inv_lre=InvertedLinearRelationalEmbedding(
            relation="city in country",
            subject_layer=4,
            object_layer=7,
            rank=25,
            object_aggregation="mean",
            u=torch.rand(768, 25),
            w_inv_vs=torch.rand(768, 25),
            w_inv_bias=torch.rand(768),
        ),
        source_layer=5,
        activations_dim=768,
        n_epochs=5,
        pl_logger=False,
    )
    assert mapping.activations_dim == 768
    assert mapping.source_layer == 5
    assert mapping.target_layer == 7
    assert mapping.mapping.weight.shape == (25, 768)
