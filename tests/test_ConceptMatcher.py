import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from multitoken_estimator.Concept import Concept
from multitoken_estimator.ConceptMatcher import ConceptMatcher, ConceptMatchQuery


def test_ConceptMatcher_query(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    concept = Concept(
        "test_concept",
        "test_relation",
        layer=10,
        vector=torch.rand(768),
    )
    conceptifier = ConceptMatcher(
        model, tokenizer, [concept], layers_matcher="transformer.h.{num}"
    )
    results = conceptifier.query("This is a test", "test")
    assert len(results) == 1
    assert concept.name in results


def test_ConceptMatcher_query_bulk(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    concept = Concept(
        "test_concept",
        "test_relation",
        layer=10,
        vector=torch.rand(768),
    )
    conceptifier = ConceptMatcher(
        model, tokenizer, [concept], layers_matcher="transformer.h.{num}"
    )
    results = conceptifier.query_bulk(
        [
            ConceptMatchQuery("This is a test", "test"),
            ConceptMatchQuery("This is another test", "test"),
        ]
    )
    assert len(results) == 2
    for result in results:
        assert concept.name in result
