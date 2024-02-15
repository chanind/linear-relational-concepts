from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from linear_relational import CausalEditor, ConceptSwapRequest, Prompt
from tokenizers import Tokenizer

from linear_relational_concepts.lib.balance_grouped_items import balance_grouped_items
from linear_relational_concepts.lib.logger import log_or_print
from linear_relational_concepts.lib.util import batchify, sample_or_all


@dataclass
class AnswerProbability:
    answer: str
    token_probabilties: list[float]

    @property
    def probability(self) -> float:
        # use min token probability to avoid penalizing longer answers
        return min(self.token_probabilties)
        # use geometric mean of token probabilities to avoid penalizing longer answers
        # return np.exp(np.log(self.token_probabilties).mean())


@dataclass
class PromptEditResult:
    original_concept: str
    prompt: Prompt
    target_concept: str
    target_answer_probabilities: list[AnswerProbability]
    original_answer_probabilities: list[AnswerProbability]

    @property
    def num_target_tokens(self) -> int:
        return min(
            len(answer_probability.token_probabilties)
            for answer_probability in self.target_answer_probabilities
        )

    @property
    def num_original_tokens(self) -> int:
        return min(
            len(answer_probability.token_probabilties)
            for answer_probability in self.original_answer_probabilities
        )

    @property
    def is_succesful(self) -> bool:
        # The edit is successful if the target answers have higher probability than the original
        max_target_prob = max(
            answer_probability.probability
            for answer_probability in self.target_answer_probabilities
        )
        max_original_prob = max(
            answer_probability.probability
            for answer_probability in self.original_answer_probabilities
        )
        return max_target_prob > max_original_prob


@dataclass
class RelationCausalityResult:
    relation: str
    prompt_edit_results: list[PromptEditResult]

    @property
    def total(self) -> int:
        return len(self.prompt_edit_results)

    @property
    def num_concepts_evaluated(self) -> int:
        return len(set(r.original_concept for r in self.prompt_edit_results))

    @property
    def total_correct(self) -> int:
        return sum(1 for r in self.prompt_edit_results if r.is_succesful)

    @property
    def causality(self) -> float:
        if self.total == 0:
            return 0
        return self.total_correct / self.total


def avg_causality(causality_results: Iterable[RelationCausalityResult]) -> float:
    causality_sum: float = 0
    causality_total = 0
    for res in causality_results:
        # if no prompts were evaluated, skip this group
        if res.total > 0:
            causality_total += 1
            causality_sum += res.causality
    if causality_total == 0:
        return 0
    return causality_sum / causality_total


def evaluate_relation_causality(
    relation_name: str,
    editor: CausalEditor,
    # these prompts are assumed to be pre-filtered to only include prompts that the model gets correct
    concept_prompts: dict[str, list[Prompt]],
    concept_vector_layer: int,
    batch_size: int = 32,
    max_swaps_per_concept: int = 5,
    max_swaps_total: Optional[int] = None,
    magnitude_multiplier: float = 1.0,
    edit_single_layer_only: bool = False,
    use_remove_concept_projection_magnitude: bool = False,
    verbose: bool = True,
) -> RelationCausalityResult:
    tokenizer = editor.tokenizer
    concepts = editor.concepts
    valid_concept_names: set[str] = set()
    for concept in concepts:
        if concept.relation != relation_name:
            raise ValueError(
                f"Concept {concept.name} is not in group {relation_name}, but in {concept.relation}"
            )
        if len(concept_prompts[concept.name]) == 0:
            log_or_print(
                f"Skipping concept {concept.name} with no prompts", verbose=verbose
            )
            continue
        valid_concept_names.add(concept.name)
    log_or_print(
        f"Valid concepts: {len(valid_concept_names)} of {len(concepts)}",
        verbose=verbose,
    )
    edits = build_causal_edits(
        concept_prompts,
        valid_concept_names,
        max_swaps_per_concept=max_swaps_per_concept,
        max_swaps_total=max_swaps_total,
    )
    prompt_edit_results: list[PromptEditResult] = []
    for edits_batch in batchify(edits, batch_size, show_progress=verbose):
        prompt_edit_results.extend(
            evaluate_causal_edits_batch(
                edits_batch,
                editor,
                tokenizer,
                concept_vector_layer=concept_vector_layer,
                magnitude_multiplier=magnitude_multiplier,
                edit_single_layer_only=edit_single_layer_only,
                use_remove_concept_projection_magnitude=use_remove_concept_projection_magnitude,
            )
        )
    group_result = RelationCausalityResult(
        relation_name,
        prompt_edit_results=prompt_edit_results,
    )
    log_or_print(
        f"Relation {relation_name} causality: {group_result.causality}",
        verbose=verbose,
    )
    return group_result


@dataclass
class EditAttempt:
    prompt: Prompt
    source_concept: str
    target_concept: str
    target_answers: tuple[str, ...]


def evaluate_causal_edits_batch(
    edits: Sequence[EditAttempt],
    editor: CausalEditor,
    tokenizer: Tokenizer,
    concept_vector_layer: int,
    magnitude_multiplier: float = 1.0,
    edit_single_layer_only: bool = False,
    use_remove_concept_projection_magnitude: bool = False,
) -> list[PromptEditResult]:
    swap_requests: list[ConceptSwapRequest] = []
    edit_results: list[PromptEditResult] = []
    # keep track of which swaps are the original preds, and which are target preds for each edit
    edits_tracking: list[tuple[list[tuple[str, int]], list[tuple[str, int]]]] = []
    for edit in edits:
        original_swaps: list[tuple[str, int]] = []
        target_swaps: list[tuple[str, int]] = []
        original_answer = edit.prompt.answer
        original_swaps.append((original_answer, len(swap_requests)))
        swap_requests.append(
            ConceptSwapRequest(
                text=combine_prompt_and_answer(edit.prompt.text, original_answer),
                subject=edit.prompt.subject,
                remove_concept=edit.source_concept,
                add_concept=edit.target_concept,
            )
        )
        for target_answer in edit.target_answers:
            target_swaps.append((target_answer, len(swap_requests)))
            swap_requests.append(
                ConceptSwapRequest(
                    text=combine_prompt_and_answer(edit.prompt.text, target_answer),
                    subject=edit.prompt.subject,
                    remove_concept=edit.source_concept,
                    add_concept=edit.target_concept,
                )
            )
        edits_tracking.append((original_swaps, target_swaps))
    swap_results = editor.swap_subject_concepts_and_predict_all_token_probs_bulk(
        swap_requests,
        magnitude_multiplier=magnitude_multiplier,
        edit_single_layer=concept_vector_layer if edit_single_layer_only else False,
        use_remove_concept_projection_magnitude=use_remove_concept_projection_magnitude,
        show_progress=False,
        # batching is handled by the caller, no need to batch here as well
        batch_size=len(swap_requests),
    )
    for index, edit in enumerate(edits):
        prompt_tokens_len = len(tokenizer.encode(edit.prompt.text.strip()))
        original_swaps, target_swaps = edits_tracking[index]
        original_answer_probabilities: list[AnswerProbability] = []
        target_answer_probabilities: list[AnswerProbability] = []
        for answer, swap_index in original_swaps:
            answer_probability = extract_answer_probability(
                all_token_probs=swap_results[swap_index],
                tokenizer=tokenizer,
                prompt_text=edit.prompt.text,
                answer=answer,
                prompt_tokens_len=prompt_tokens_len,
            )
            original_answer_probabilities.append(answer_probability)
        for answer, swap_index in target_swaps:
            answer_probability = extract_answer_probability(
                all_token_probs=swap_results[swap_index],
                tokenizer=tokenizer,
                prompt_text=edit.prompt.text,
                answer=answer,
                prompt_tokens_len=prompt_tokens_len,
            )
            target_answer_probabilities.append(answer_probability)
        edit_results.append(
            PromptEditResult(
                original_concept=edit.source_concept,
                prompt=edit.prompt,
                target_concept=edit.target_concept,
                target_answer_probabilities=target_answer_probabilities,
                original_answer_probabilities=original_answer_probabilities,
            )
        )
    return edit_results


def extract_answer_probability(
    all_token_probs: torch.Tensor,
    tokenizer: Tokenizer,
    prompt_text: str,
    answer: str,
    prompt_tokens_len: int,
) -> AnswerProbability:
    full_tokens = tokenizer.encode(
        combine_prompt_and_answer(prompt_text, answer),
        return_tensors="pt",
    ).squeeze()
    # each token should be the prediction of the previous token, so we drop the first index
    # we also don't care about the final token, we just care about predicted final token, which comes
    # from the second to last index. We can/should just strip the final token from the prompt in the future
    full_tokens_shifted = full_tokens[1:]
    specific_token_probs = all_token_probs[
        torch.arange(full_tokens_shifted.shape[-1]), full_tokens_shifted
    ]
    token_probabilities = specific_token_probs[prompt_tokens_len - 1 :].tolist()
    if len(token_probabilities) == 0:
        raise ValueError(
            f"Unable to determine answer position for '{answer}' in prompt '{prompt_text}' with tokens {full_tokens.tolist()}"
        )
    return AnswerProbability(
        answer=answer,
        token_probabilties=token_probabilities,
    )


def combine_prompt_and_answer(prompt: str, answer: str) -> str:
    return prompt.strip() + " " + answer.strip()


def build_causal_edits(
    concept_prompts: dict[str, list[Prompt]],
    valid_concept_names: set[str],
    max_swaps_per_concept: int = 5,
    max_swaps_total: Optional[int] = None,
) -> list[EditAttempt]:
    edits_to_attempt_per_concept: dict[str, list[EditAttempt]] = defaultdict(list)
    for concept_name in valid_concept_names:
        remaining_concepts = valid_concept_names - {concept_name}
        for target_concept in remaining_concepts:
            target_concept_prompts = concept_prompts[target_concept]
            all_target_expected_answers: set[str] = set()
            # sample here to avoid potential combinatorial explosion of edits
            for prompt in sample_or_all(
                target_concept_prompts, k=max_swaps_per_concept
            ):
                all_target_expected_answers.add(prompt.answer)
                # sample here to avoid potential combinatorial explosion of edits
                for source_prompt in sample_or_all(
                    concept_prompts[concept_name], k=max_swaps_per_concept
                ):
                    edits_to_attempt_per_concept[concept_name].append(
                        EditAttempt(
                            prompt=source_prompt,
                            source_concept=concept_name,
                            target_concept=target_concept,
                            target_answers=tuple(all_target_expected_answers),
                        )
                    )
    edits_to_attempt = balance_grouped_items(
        edits_to_attempt_per_concept,
        max_total=max_swaps_total,
        max_per_group=max_swaps_per_concept,
    )
    return edits_to_attempt
