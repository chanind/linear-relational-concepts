import random
from dataclasses import dataclass
from typing import Callable

from multitoken_estimator.data_model import EntityDataModel, SampleDataModel
from multitoken_estimator.database import Database

EntityModifier = Callable[[str], str]

original_entry: EntityModifier = lambda s: s
uppercase_entry: EntityModifier = lambda s: s.upper()
lowercase_entry: EntityModifier = lambda s: s.lower()
split_entry_with_dashes: EntityModifier = lambda s: "-".join(s).replace("- -", " ")
split_entry_with_periods: EntityModifier = lambda s: ".".join(s).replace(". .", " ")
split_entry_with_spaces: EntityModifier = lambda s: " ".join(s).replace("   ", " ")

DEFAULT_ENTITY_MODIFIERS = [
    original_entry,
    uppercase_entry,
    lowercase_entry,
    split_entry_with_dashes,
    split_entry_with_periods,
    split_entry_with_spaces,
]


@dataclass(frozen=True, slots=True)
class Prompt:
    text: str
    answer: str
    subject: str


class PromptGenerator:
    """
    Generate prompts which match various criteria from the database.
    """

    db: Database

    def __init__(self, db: Database) -> None:
        self.db = db

    def generate_prompts_for_object(
        self,
        object_name: str,
        num_fsl_examples: int = 5,
        entity_modifiers: list[EntityModifier] = DEFAULT_ENTITY_MODIFIERS,
        exclude_fsl_examples_of_object: bool = True,
    ) -> set[Prompt]:
        object = self.db.query_one_or_throw(
            EntityDataModel, lambda e: e.name == object_name
        )
        object_samples = self.db.query_all(
            SampleDataModel, lambda s: s.object == object
        )
        relations = {s.relation for s in object_samples}
        samples_by_relation_id = {}
        for relation in relations:
            selector = lambda s: s.relation == relation
            if exclude_fsl_examples_of_object:
                selector = lambda s: s.relation == relation and s.object != object
            samples_by_relation_id[relation.id] = self.db.query_all(
                SampleDataModel, selector
            )

        prompts: set[Prompt] = set()
        for object_sample in object_samples:
            answers = {object.name}
            if object.alternative_names is not None:
                answers.update(object.alternative_names)
            for answer in answers:
                for entity_modifier in entity_modifiers:
                    for template in object_sample.relation.templates:
                        text = format_prompt_text(
                            template=template,
                            sample=object_sample,
                            potential_fsl_samples=samples_by_relation_id[
                                object_sample.relation.id
                            ],
                            object_modifier=entity_modifier,
                            num_fsl_examples=num_fsl_examples,
                        )
                        prompts.add(
                            Prompt(
                                text=text,
                                answer=entity_modifier(answer),
                                subject=object_sample.subject.name,
                            )
                        )
        return prompts


def format_prompt_text(
    template: str,
    sample: SampleDataModel,
    potential_fsl_samples: list[SampleDataModel],
    subject_modifier: EntityModifier | None = None,
    object_modifier: EntityModifier | None = None,
    num_fsl_examples: int = 5,
) -> str:
    """
    Format a prompt text for a given sample and a list of FSL samples.
    """
    fsl_samples = random.sample(
        potential_fsl_samples, min(num_fsl_examples, len(potential_fsl_samples))
    )
    texts = []
    for fsl_sample in fsl_samples:
        object = fsl_sample.object.name
        subject = fsl_sample.subject.name
        if object_modifier is not None:
            object = object_modifier(object)
        if subject_modifier is not None:
            subject = subject_modifier(subject)
        texts.append(template.format(subject=subject).strip() + " " + object.strip())
    subject = sample.subject.name
    if subject_modifier is not None:
        subject = subject_modifier(subject)
    texts.append(template.format(subject=subject).strip())
    return "\n".join(texts)
