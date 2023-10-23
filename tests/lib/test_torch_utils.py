from transformers import GPT2LMHeadModel

from multitoken_estimator.lib.torch_utils import guess_model_name


def test_guess_model_name(model: GPT2LMHeadModel) -> None:
    assert guess_model_name(model) == "gpt2"
