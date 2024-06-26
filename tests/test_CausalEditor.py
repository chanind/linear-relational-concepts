from linear_relational import CausalEditor, ConceptSwapAndPredictGreedyRequest
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from tests.helpers import quick_concept


def test_CausalEditor_swap_subject_concepts_and_predict_greedy_single(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    china = quick_concept(model, tokenizer, "Beijing")
    japan = quick_concept(model, tokenizer, "Tokyo")

    editor = CausalEditor(
        model, tokenizer, [china, japan], layer_matcher="transformer.h.{num}"
    )
    result = editor.swap_subject_concepts_and_predict_greedy(
        text="Beijing is located in the country of",
        subject="Beijing",
        remove_concept=china.name,
        add_concept=japan.name,
        edit_single_layer=8,
    )
    assert result == " Japan"


def test_CausalEditor_swap_subject_concepts_and_predict_greedy_bulk(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    china = quick_concept(model, tokenizer, "Beijing")
    japan = quick_concept(model, tokenizer, "Tokyo")
    germany = quick_concept(model, tokenizer, "Berlin")

    editor = CausalEditor(
        model, tokenizer, [china, japan, germany], layer_matcher="transformer.h.{num}"
    )
    results = editor.swap_subject_concepts_and_predict_greedy_bulk(
        [
            ConceptSwapAndPredictGreedyRequest(
                text="Beijing is located in the country of",
                subject="Beijing",
                remove_concept=china.name,
                add_concept=japan.name,
            ),
            ConceptSwapAndPredictGreedyRequest(
                text="Berlin is located in the country of",
                subject="Berlin",
                remove_concept=germany.name,
                add_concept=china.name,
            ),
            ConceptSwapAndPredictGreedyRequest(
                text="Tokyo is located in the country of",
                subject="Tokyo",
                remove_concept=japan.name,
                add_concept=germany.name,
            ),
        ],
        edit_single_layer=8,
        magnitude_multiplier=1.0,
    )
    assert results == [" Japan", " China", " Germany"]
