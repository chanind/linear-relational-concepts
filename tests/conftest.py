import pytest
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# loading in advance so it won't reload on every test
# just need to make sure not to edit these models in tests...
_model = GPT2LMHeadModel.from_pretrained("gpt2")
_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


@pytest.fixture
def model() -> GPT2LMHeadModel:
    return _model


@pytest.fixture
def tokenizer() -> GPT2TokenizerFast:
    return _tokenizer
