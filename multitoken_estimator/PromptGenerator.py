from dataclasses import dataclass
from typing import Callable, Optional

from multitoken_estimator.data.RelationDataset import RelationDataset, Sample
from multitoken_estimator.lib.logger import logger
from multitoken_estimator.lib.util import dedupe_stable, stable_sample

EntityModifier = Callable[[str], str]


def chain_modifiers(modifiers: list[EntityModifier]) -> EntityModifier:
    def chained_modifier(s: str) -> str:
        for modifier in modifiers:
            s = modifier(s)
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

AUGMENTATION_MODIFIERS = [
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
    relation_name: str


@dataclass
class PromptGenerator:
    """
    Generate prompts which match various criteria from the database.
    """

    db: RelationDataset
    seed: int | float | str = 42

    def generate_prompts_for_all_relations(
        self,
        num_fsl_examples: int = 5,
        entity_modifiers: list[EntityModifier] | None = None,
        exclude_fsl_examples_of_object: bool = True,
    ) -> list[Prompt]:
        relation_names = {rel.name for rel in self.db.relations}
        prompts: list[Prompt] = []
        for relation_name in relation_names:
            prompts.extend(
                self.generate_prompts_for_relation(
                    relation_name=relation_name,
                    num_fsl_examples=num_fsl_examples,
                    entity_modifiers=entity_modifiers,
                    exclude_fsl_examples_of_object=exclude_fsl_examples_of_object,
                )
            )
        return prompts

    def generate_prompts_for_relation(
        self,
        relation_name: str,
        num_fsl_examples: int = 5,
        entity_modifiers: list[EntityModifier] | None = None,
        exclude_fsl_examples_of_object: bool = True,
    ) -> list[Prompt]:
        samples_in_relation = self.db.get_relation_samples(relation_name)
        object_names = {sample.object for sample in samples_in_relation}
        prompts: list[Prompt] = []
        for object_name in object_names:
            prompts.extend(
                self.generate_prompts_for_object(
                    relation_name=relation_name,
                    object_name=object_name,
                    num_fsl_examples=num_fsl_examples,
                    entity_modifiers=entity_modifiers,
                    exclude_fsl_examples_of_object=exclude_fsl_examples_of_object,
                    valid_relation_names={relation_name},
                )
            )
        return prompts

    def generate_prompts_for_object(
        self,
        relation_name: str,
        object_name: str,
        num_fsl_examples: int = 5,
        entity_modifiers: list[EntityModifier] | None = None,
        exclude_fsl_examples_of_object: bool = True,
        valid_relation_names: Optional[set[str]] = None,
    ) -> list[Prompt]:
        object_samples = self.db.get_object_samples(
            relation=relation_name, object=object_name
        )
        relation = self.db.get_relation(relation_name)
        if valid_relation_names is not None:
            object_samples = [
                s for s in object_samples if s.relation in valid_relation_names
            ]
        relation_samples = self.db.get_relation_samples(relation_name)

        prompts: list[Prompt] = []
        for object_sample in object_samples:
            answer = object_name
            potential_fsl_samples = relation_samples
            if exclude_fsl_examples_of_object:
                potential_fsl_samples = [
                    s for s in potential_fsl_samples if s.object != object_name
                ]

            if num_fsl_examples > 0 and len(potential_fsl_samples) == 0:
                logger.warn(
                    f"Warning: no FSL samples for relation {object_sample.relation} and object {object_name}",
                )
            for entity_modifier in entity_modifiers or [null_modifier]:
                for template in relation.templates:
                    text = format_prompt_text(
                        template=template,
                        sample=object_sample,
                        potential_fsl_samples=potential_fsl_samples,
                        object_modifier=entity_modifier,
                        num_fsl_examples=num_fsl_examples,
                        seed=self.seed,
                    )
                    prompts.append(
                        Prompt(
                            text=text,
                            answer=entity_modifier(answer),
                            subject=object_sample.subject,
                            object_name=object_name,
                            relation_name=relation_name,
                        )
                    )
        return dedupe_stable(prompts)


def format_prompt_text(
    template: str,
    sample: Sample,
    potential_fsl_samples: list[Sample],
    subject_modifier: EntityModifier | None = None,
    object_modifier: EntityModifier | None = None,
    num_fsl_examples: int = 5,
    seed: int | float | str = 42,
) -> str:
    """
    Format a prompt text for a given sample and a list of FSL samples.
    """
    fsl_samples = stable_sample(
        potential_fsl_samples,
        min(num_fsl_examples, len(potential_fsl_samples)),
        f"{sample.relation}-{sample.object}-{sample.subject}-{seed}",
    )
    texts = []
    for fsl_sample in fsl_samples:
        object = fsl_sample.object
        subject = fsl_sample.subject
        if object_modifier is not None:
            object = object_modifier(object)
        if subject_modifier is not None:
            subject = subject_modifier(subject)
        texts.append(template.format(subject).strip() + " " + object.strip())
    subject = sample.subject
    if subject_modifier is not None:
        subject = subject_modifier(subject)
    texts.append(template.format(subject).strip())
    return "\n".join(texts)
