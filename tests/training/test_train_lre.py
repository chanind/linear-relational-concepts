from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from multitoken_estimator.PromptGenerator import Prompt
from multitoken_estimator.training.train_lre import train_lre


def test_train_lre(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast) -> None:
    fsl_prefixes = "\n".join(
        [
            "Berlin is located in the country of Germany",
            "Toronto is located in the country of Canada",
            "Lagos is located in the country of Nigeria",
        ]
    )
    prompts = [
        Prompt(
            text=f"{fsl_prefixes}\nTokyo is located in the country of",
            answer="Japan",
            subject="Tokyo",
            object_name="Japan",
        ),
        Prompt(
            text=f"{fsl_prefixes}\nRome is located in the country of",
            answer="Italy",
            subject="Rome",
            object_name="Italy",
        ),
    ]
    lre = train_lre(
        model=model,
        tokenizer=tokenizer,
        layer_matcher="transformer.h.{num}",
        relation_name="city in country",
        subject_layer=5,
        object_layer=9,
        prompts=prompts,
        object_aggregation="mean",
    )
    assert lre.relation == "city in country"
    assert lre.subject_layer == 5
    assert lre.object_layer == 9
    assert lre.weight.shape == (768, 768)
    assert lre.bias.shape == (1, 768)
