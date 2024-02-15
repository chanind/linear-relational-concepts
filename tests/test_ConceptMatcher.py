import torch
from linear_relational import Concept, ConceptMatcher, ConceptMatchQuery
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def test_ConceptMatcher_query(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    concept = Concept(
        object="test_concept",
        relation="test_relation",
        layer=10,
        vector=torch.rand(768),
    )
    conceptifier = ConceptMatcher(
        model, tokenizer, [concept], layer_matcher="transformer.h.{num}"
    )
    results = conceptifier.query("This is a test", "test")
    assert len(results.concept_results) == 1
    assert concept.name in results.concept_results


def test_ConceptMatcher_query_bulk(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    concept = Concept(
        object="test_concept",
        relation="test_relation",
        layer=10,
        vector=torch.rand(768),
    )
    conceptifier = ConceptMatcher(
        model, tokenizer, [concept], layer_matcher="transformer.h.{num}"
    )
    results = conceptifier.query_bulk(
        [
            ConceptMatchQuery("This is a test", "test"),
            ConceptMatchQuery("This is another test", "test"),
        ]
    )
    assert len(results) == 2
    for result in results:
        assert concept.name in result.concept_results
