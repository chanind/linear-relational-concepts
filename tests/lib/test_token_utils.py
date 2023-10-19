import pytest
from transformers import GPT2TokenizerFast, LlamaTokenizer

from multitoken_estimator.lib.token_utils import (
    decode_tokens,
    find_final_subject_token_index,
    find_token_range,
)


def test_decode_tokens(tokenizer: GPT2TokenizerFast) -> None:
    tokens = tokenizer.encode("Hello, my dog is cute")
    decoded = decode_tokens(tokenizer, tokens)
    assert decoded == ["Hello", ",", " my", " dog", " is", " cute"]


def test_find_token_range(tokenizer: GPT2TokenizerFast) -> None:
    tokens = tokenizer.encode("Hello, my dog is cute")
    assert find_token_range(tokenizer, tokens, "dog") == (3, 4)


def test_find_token_range_returns_the_final_instance_of_the_token_by_default(
    tokenizer: GPT2TokenizerFast,
) -> None:
    tokens = tokenizer.encode("Hello, my dog is a cute dog")
    assert find_token_range(tokenizer, tokens, "dog") == (7, 8)


def test_find_token_range_returns_the_first_instance_of_the_token_if_requested(
    tokenizer: GPT2TokenizerFast,
) -> None:
    tokens = tokenizer.encode("Hello, my dog is a cute dog")
    assert find_token_range(tokenizer, tokens, "dog", find_last_match=False) == (3, 4)


def test_find_token_range_spanning_multiple_tokens(
    tokenizer: GPT2TokenizerFast,
) -> None:
    tokens = tokenizer.encode("Hello, my dog is cute")
    assert find_token_range(tokenizer, tokens, "dog is") == (3, 5)


def test_find_token_range_non_alphanum(tokenizer: GPT2TokenizerFast) -> None:
    tokens = tokenizer.encode("Jean Pierre Joseph d’Arcet has the profession of")
    assert find_token_range(tokenizer, tokens, "Jean Pierre Joseph d’Arcet") == (0, 8)


def test_find_token_range_accented(tokenizer: GPT2TokenizerFast) -> None:
    tokens = tokenizer.encode("I think Örebro is in the continent of")
    assert find_token_range(tokenizer, tokens, "Örebro") == (2, 5)


def test_find_token_missing_token(tokenizer: GPT2TokenizerFast) -> None:
    tokens = tokenizer.encode("Hello, my dog is cute")
    with pytest.raises(ValueError):
        find_token_range(tokenizer, tokens, "cat")


def test_find_token_works_with_llama(vicuna_tokenizer: LlamaTokenizer) -> None:
    tokens = vicuna_tokenizer.encode("I think Paris is located in the country of")
    assert find_token_range(vicuna_tokenizer, tokens, "Paris") == (3, 4)


def test_find_final_subject_token_index(tokenizer: GPT2TokenizerFast) -> None:
    assert (
        find_final_subject_token_index(
            tokenizer, "Bill Gates is the CEO of", "Bill Gates"
        )
        == 1
    )
    assert (
        find_final_subject_token_index(tokenizer, "Bill Gates is the CEO of", "Bill")
        == 0
    )
