from typing import Optional

import torch
from linear_relational import Concept, Prompt
from linear_relational.lib.extract_token_activations import extract_token_activations
from linear_relational.lib.token_utils import find_final_word_token_index
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def create_prompt(subject: str, answer: str, text: Optional[str] = None) -> Prompt:
    return Prompt(
        text=text or f"{subject} is located in the country of",
        answer=answer,
        subject=subject,
        object_name=answer,
    )


def quick_concept(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    subject: str,
    relation: str = "test group",
    object: Optional[str] = None,
    layer: int = 8,
) -> Concept:
    """
    Hacky way to get a concept for a subject by just using norm raw activation of the subject as the concept vector.
    """
    raw_vec = extract_token_activations(
        model,
        tokenizer,
        ["transformer.h.8"],
        texts=[subject],
        token_indices=[find_final_word_token_index(tokenizer, subject, subject)],
    )[0]["transformer.h.8"][0]
    concept_vec = raw_vec / torch.norm(raw_vec)
    return Concept(
        object=(object or subject),
        relation=relation,
        vector=concept_vec,
        layer=layer,
    )
