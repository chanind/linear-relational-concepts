import random
from dataclasses import dataclass
from typing import Callable, Optional

from multitoken_estimator.data_model import EntityDataModel, SampleDataModel
from multitoken_estimator.database import Database

EntityModifier = Callable[[str], str]


def chain_modifiers(modifiers: list[EntityModifier]) -> EntityModifier:
    def chained_modifier(s: str) -> str:
        for m in modifiers:
            s = m(s)
        return s

    return chained_modifier


null_modifier: EntityModifier = lambda s: s
uppercase_modifier: EntityModifier = lambda s: s.upper()
lowercase_modifier: EntityModifier = lambda s: s.lower()
split_with_dashes_modifier: EntityModifier = lambda s: "-".join(s).replace("- -", " ")
split_with_periods_modifier: EntityModifier = lambda s: ".".join(s).replace(". .", " ")
split_with_spaces_modifier: EntityModifier = lambda s: " ".join(s).replace("   ", " ")
split_with_periods_uppercase_modifier = chain_modifiers(
    [uppercase_modifier, split_with_periods_modifier]
)
split_with_dashes_uppercase_modifier = chain_modifiers(
    [uppercase_modifier, split_with_dashes_modifier]
)

DEFAULT_ENTITY_MODIFIERS = [
    null_modifier,
    uppercase_modifier,
    lowercase_modifier,
    split_with_spaces_modifier,
    split_with_periods_uppercase_modifier,
    split_with_dashes_uppercase_modifier,
]


@dataclass(frozen=True, slots=True)
class Prompt:
    text: str
    answer: str
    subject: str
    object_name: str


class PromptGenerator:
    """
    Generate prompts which match various criteria from the database.
    """

    db: Database

    def __init__(self, db: Database) -> None:
        self.db = db

    def generate_prompts_for_relation(
        self,
        relation_name: str,
        num_fsl_examples: int = 5,
        entity_modifiers: list[EntityModifier] | None = DEFAULT_ENTITY_MODIFIERS,
        exclude_fsl_examples_of_object: bool = True,
    ) -> set[Prompt]:
        samples_in_relation = self.db.query_all(
            SampleDataModel, lambda s: s.relation.name == relation_name
        )
        object_names = {s.object.name for s in samples_in_relation}
        prompts: set[Prompt] = set()
        for object_name in object_names:
            prompts.update(
                self.generate_prompts_for_object(
                    object_name=object_name,
                    num_fsl_examples=num_fsl_examples,
                    entity_modifiers=entity_modifiers,
                    exclude_fsl_examples_of_object=exclude_fsl_examples_of_object,
                    valid_relations={relation_name},
                )
            )
        return prompts

    def generate_prompts_for_object(
        self,
        object_name: str,
        num_fsl_examples: int = 5,
        entity_modifiers: list[EntityModifier] | None = DEFAULT_ENTITY_MODIFIERS,
        exclude_fsl_examples_of_object: bool = True,
        valid_relations: Optional[set[str]] = None,
    ) -> set[Prompt]:
        object = self.db.query_one_or_throw(
            EntityDataModel, lambda e: e.name == object_name
        )
        object_samples = self.db.query_all(
            SampleDataModel, lambda s: s.object == object
        )
        relations = {s.relation for s in object_samples}
        if valid_relations is not None:
            relations = relations.intersection(valid_relations)
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
                for entity_modifier in entity_modifiers or [null_modifier]:
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
                                object_name=object_name,
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
        texts.append(template.format(subject).strip() + " " + object.strip())
    subject = sample.subject.name
    if subject_modifier is not None:
        subject = subject_modifier(subject)
    texts.append(template.format(subject).strip())
    return "\n".join(texts)
