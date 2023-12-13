from collections import defaultdict

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from linear_relational_concepts.CausalEditor import CausalEditor
from linear_relational_concepts.evaluation.evaluate_relation_causality import (
    EditAttempt,
    build_causal_edits,
    evaluate_causal_edits_batch,
)
from tests.helpers import create_prompt, quick_concept


def test_evaluate_causal_edits_batch_simple(
    model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer
) -> None:
    china = quick_concept(model, tokenizer, "Beijing")
    japan = quick_concept(model, tokenizer, "Tokyo")

    editor = CausalEditor(
        model, tokenizer, [china, japan], layer_matcher="transformer.h.{num}"
    )

    edits = [
        EditAttempt(
            prompt=create_prompt(
                text="Beijing is located in the country of",
                subject="Beijing",
                answer="China",
            ),
            source_concept=china.name,
            target_concept=japan.name,
            target_answers=("Japan",),
        ),
        EditAttempt(
            prompt=create_prompt(
                text="Tokyo is located in the country of",
                subject="Tokyo",
                answer="Japan",
            ),
            source_concept=japan.name,
            target_concept=china.name,
            target_answers=("China",),
        ),
    ]

    results = evaluate_causal_edits_batch(
        edits,
        editor=editor,
        tokenizer=tokenizer,
        concept_vector_layer=8,
        magnitude_multiplier=1.0,
        edit_single_layer_only=True,
    )
    assert len(results) == 2
    for result in results:
        assert len(result.original_answer_probabilities) == 1
        assert len(result.target_answer_probabilities) == 1
        assert len(result.original_answer_probabilities[0].token_probabilties) == 1
        assert len(result.target_answer_probabilities[0].token_probabilties) == 1
        assert result.is_succesful


def test_evaluate_causal_edits_batch_with_multitoken_answers(
    model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer
) -> None:
    china = quick_concept(model, tokenizer, "Beijing")
    japan = quick_concept(model, tokenizer, "Tokyo")

    editor = CausalEditor(
        model, tokenizer, [china, japan], layer_matcher="transformer.h.{num}"
    )

    edits = [
        EditAttempt(
            prompt=create_prompt(
                text="Beijing is located in the country of",
                subject="Beijing",
                answer="China.",
            ),
            source_concept=china.name,
            target_concept=japan.name,
            target_answers=("Japan.",),
        ),
        EditAttempt(
            prompt=create_prompt(
                text="Tokyo is located in the country of",
                subject="Tokyo",
                answer="Japan.",
            ),
            source_concept=japan.name,
            target_concept=china.name,
            target_answers=("China.",),
        ),
    ]

    results = evaluate_causal_edits_batch(
        edits,
        editor=editor,
        tokenizer=tokenizer,
        concept_vector_layer=8,
        magnitude_multiplier=1.0,
        edit_single_layer_only=True,
    )
    assert len(results) == 2
    for result in results:
        assert len(result.original_answer_probabilities) == 1
        assert len(result.target_answer_probabilities) == 1
        assert len(result.original_answer_probabilities[0].token_probabilties) == 2
        assert len(result.target_answer_probabilities[0].token_probabilties) == 2
        assert result.is_succesful


def test_evaluate_causal_edits_batch_with_unsuccessful_causality(
    model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer
) -> None:
    china = quick_concept(model, tokenizer, "Beijing")
    japan = quick_concept(model, tokenizer, "Tokyo")

    editor = CausalEditor(
        model, tokenizer, [china, japan], layer_matcher="transformer.h.{num}"
    )

    edits = [
        EditAttempt(
            prompt=create_prompt(
                text="Beijing is located in the country of",
                subject="Beijing",
                answer="China.",
            ),
            source_concept=china.name,
            target_concept=japan.name,
            target_answers=("Japan.",),
        ),
        EditAttempt(
            prompt=create_prompt(
                text="Tokyo is located in the country of",
                subject="Tokyo",
                answer="Japan.",
            ),
            source_concept=japan.name,
            target_concept=china.name,
            target_answers=("China.",),
        ),
    ]

    results = evaluate_causal_edits_batch(
        edits,
        editor=editor,
        tokenizer=tokenizer,
        concept_vector_layer=8,
        magnitude_multiplier=0.1,  # 0.1 is too small to cause a change
        edit_single_layer_only=True,
    )
    for result in results:
        assert not result.is_succesful


def test_build_causal_edits() -> None:
    valid_concept_names = {"c1", "c2", "c3"}
    concept_prompts = {
        "c1": [
            create_prompt("c1subj1", "c1ans1"),
        ],
        "c2": [
            create_prompt("c2subj1", "c2ans1"),
        ],
        "c3": [
            create_prompt("c3subj1", "c3ans1"),
        ],
    }

    results = build_causal_edits(
        concept_prompts,
        valid_concept_names,
    )

    assert len(results) == 6

    concept_edits = {(edit.source_concept, edit.target_concept) for edit in results}
    assert concept_edits == {
        ("c1", "c2"),
        ("c1", "c3"),
        ("c2", "c1"),
        ("c2", "c3"),
        ("c3", "c1"),
        ("c3", "c2"),
    }
    for edit in results:
        assert edit.prompt == concept_prompts[edit.source_concept][0]


def test_build_causal_edits_limits_swaps_per_concept() -> None:
    valid_concept_names = {"c1", "c2", "c3"}
    concept_prompts = {
        "c1": [
            create_prompt("c1subj1", "c1ans1"),
            create_prompt("c1subj2", "c1ans2"),
        ],
        "c2": [
            create_prompt("c2subj1", "c2ans1"),
            create_prompt("c2subj2", "c2ans2"),
        ],
        "c3": [
            create_prompt("c3subj1", "c3ans1"),
            create_prompt("c3subj2", "c3ans2"),
        ],
    }

    results = build_causal_edits(
        concept_prompts,
        valid_concept_names,
        max_swaps_per_concept=5,
    )

    assert len(results) == 15

    source_concept_edit_counts: dict[str, int] = defaultdict(int)
    for edit in results:
        source_concept_edit_counts[edit.source_concept] += 1
    assert source_concept_edit_counts == {"c1": 5, "c2": 5, "c3": 5}


def test_build_causal_edits_can_limit_total_swaps() -> None:
    valid_concept_names = {"c1", "c2", "c3"}
    concept_prompts = {
        "c1": [
            create_prompt("c1subj1", "c1ans1"),
            create_prompt("c1subj2", "c1ans2"),
        ],
        "c2": [
            create_prompt("c2subj1", "c2ans1"),
            create_prompt("c2subj2", "c2ans2"),
        ],
        "c3": [
            create_prompt("c3subj1", "c3ans1"),
            create_prompt("c3subj2", "c3ans2"),
        ],
    }

    results = build_causal_edits(
        concept_prompts,
        valid_concept_names,
        max_swaps_total=7,
    )

    assert len(results) == 7

    source_concept_edit_counts: dict[str, int] = defaultdict(int)
    for edit in results:
        source_concept_edit_counts[edit.source_concept] += 1
    # each concept should get roughtly the same number of edits
    assert set(source_concept_edit_counts.keys()) == {"c1", "c2", "c3"}
    for count in source_concept_edit_counts.values():
        assert count >= 2
        assert count <= 3
