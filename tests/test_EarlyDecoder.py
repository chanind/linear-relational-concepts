from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from linear_relational_concepts.EarlyDecoder import EarlyDecoder


def test_EarlyDecoder(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast) -> None:
    decoder = EarlyDecoder(
        model, tokenizer, decoding_layers=["transformer.ln_f", "lm_head"]
    )
    dog_token = tokenizer.encode("dog")[0]
    dog_activation = model.transformer.wte.weight[dog_token]
    top_decodings = decoder.decode(dog_activation, topk=3)
    assert top_decodings == ["dog", "dogs", "Dog"]
