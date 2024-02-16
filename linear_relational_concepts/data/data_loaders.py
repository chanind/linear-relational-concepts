from dataclasses import dataclass
from typing import Optional

from dataclasses_json import DataClassJsonMixin
from linear_relational.lib.util import dedupe_stable

from linear_relational_concepts.data.RelationDataset import (
    Relation,
    RelationDataset,
    Sample,
)
from linear_relational_concepts.lib.constants import DATA_DIR


@dataclass
class LreProperties(DataClassJsonMixin):
    relation_type: str
    domain_name: str
    range_name: str
    symmetric: bool


@dataclass
class LreSample(DataClassJsonMixin):
    subject: str
    object: str


@dataclass
class LreRelation(DataClassJsonMixin):
    name: str
    prompt_templates: list[str]
    prompt_templates_zs: list[str] | None
    properties: LreProperties
    samples: list[LreSample]


def lre_relation_to_relation_and_samples(
    lre_relation: LreRelation,
) -> tuple[Relation, list[Sample]]:
    relation = Relation(
        name=lre_relation.name,
        templates=tuple(lre_relation.prompt_templates),
        zs_templates=(
            tuple(lre_relation.prompt_templates_zs)
            if lre_relation.prompt_templates_zs is not None
            else None
        ),
        category=lre_relation.properties.relation_type,
    )

    samples = dedupe_stable(
        [
            Sample(
                relation=lre_relation.name,
                subject=sample.subject,
                object=sample.object,
            )
            for sample in lre_relation.samples
        ]
    )
    return relation, samples


def load_lre_data(only_load_files: Optional[set[str]] = None) -> RelationDataset:
    db = RelationDataset()
    for relation_file in DATA_DIR.glob("lre/*/*.json"):
        if only_load_files is not None and relation_file.name not in only_load_files:
            continue
        with open(relation_file, encoding="utf-8") as f:
            raw_json_str = f.read()
            # some of the JSON encoding is messed up, need to replace these double backslashes
            # e.g. Bogot\\u00e1 instead of Bogot\u00e1 (Bogotá)
            lre_relation = LreRelation.from_json(raw_json_str.replace("\\\\u", "\\u"))
            relation, samples = lre_relation_to_relation_and_samples(lre_relation)
        db.add_relation(relation)
        for sample in samples:
            db.add_sample(sample)
    return db
