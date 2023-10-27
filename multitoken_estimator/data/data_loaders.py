from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dataclasses_json import DataClassJsonMixin

from multitoken_estimator.data.RelationDataset import Relation, RelationDataset, Sample
from multitoken_estimator.lib.constants import DATA_DIR


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
    prompt_templates_zs: list[str]
    properties: LreProperties
    samples: list[LreSample]


def lre_relation_to_relation_and_samples(
    lre_relation: LreRelation,
) -> tuple[Relation, list[Sample]]:
    relation_to_lre_type_map = get_relation_to_lre_type_map()

    relation = Relation(
        name=lre_relation.name,
        templates=frozenset(lre_relation.prompt_templates),
        zs_templates=frozenset(lre_relation.prompt_templates_zs),
        category=relation_to_lre_type_map.get(lre_relation.name),
    )

    samples = [
        Sample(
            relation=lre_relation.name,
            subject=sample.subject,
            object=sample.object,
        )
        for sample in lre_relation.samples
    ]
    return relation, samples


def load_lre_data(only_load_files: Optional[set[str]] = None) -> RelationDataset:
    db = RelationDataset()
    for relation_file in DATA_DIR.glob("lre/*/*.json"):
        if only_load_files is not None and relation_file.name not in only_load_files:
            continue
        with open(relation_file) as f:
            lre_relation = LreRelation.from_json(f.read())
            relation, samples = lre_relation_to_relation_and_samples(lre_relation)
        db.add_relation(relation)
        for sample in samples:
            db.add_sample(sample)
    return db


@lru_cache(maxsize=1)
def get_relation_to_lre_type_map() -> dict[str, str]:
    """
    map relation name to LRE type
    """
    relation_to_lre_type_map = {}
    for relation_file in DATA_DIR.glob("lre/*/*.json"):
        lre_type = relation_file.parent.name
        with open(relation_file) as f:
            lre_relation = LreRelation.from_json(f.read())
        relation_to_lre_type_map[lre_relation.name] = lre_type
    return relation_to_lre_type_map
