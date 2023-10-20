from typing import Optional

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from multitoken_estimator.Concept import Concept
from multitoken_estimator.lib.extract_token_activations import extract_token_activations
from multitoken_estimator.lib.token_utils import find_final_subject_token_index


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
        token_indices=[find_final_subject_token_index(tokenizer, subject, subject)],
    )[0]["transformer.h.8"][0]
    concept_vec = raw_vec / torch.norm(raw_vec)
    return Concept(
        object=(object or subject),
        relation=relation,
        vector=concept_vec,
        layer=layer,
    )
