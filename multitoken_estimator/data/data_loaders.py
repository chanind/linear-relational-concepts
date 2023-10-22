from functools import lru_cache
from typing import Optional

from multitoken_estimator.data.data_model import (
    Entity,
    LreRelation,
    RelationData,
    RelationSample,
)
from multitoken_estimator.data.database import Database
from multitoken_estimator.lib.constants import DATA_DIR


def lre_relation_to_relation_data(lre_relation: LreRelation) -> RelationData:
    relation_to_lre_type_map = get_relation_to_lre_type_map()

    return RelationData(
        name=lre_relation.name,
        templates=frozenset(lre_relation.prompt_templates),
        zs_templates=frozenset(lre_relation.prompt_templates_zs),
        samples=[
            RelationSample(
                subject=Entity(
                    name=sample.subject, type=lre_relation.properties.domain_name
                ),
                object=Entity(
                    name=sample.object, type=lre_relation.properties.range_name
                ),
            )
            for sample in lre_relation.samples
        ],
        category=relation_to_lre_type_map.get(lre_relation.name),
    )


def load_lre_data(only_load_files: Optional[set[str]] = None) -> Database:
    db = Database()
    for relation_file in DATA_DIR.glob("lre/*/*.json"):
        if only_load_files is not None and relation_file.name not in only_load_files:
            continue
        with open(relation_file) as f:
            lre_relation = LreRelation.from_json(f.read())
            relation_data = lre_relation_to_relation_data(lre_relation)
        db.add_relation_data(relation_data)
    return db


def load_custom_data() -> Database:
    db = Database()
    for relation_file in DATA_DIR.glob("custom/*.json"):
        with open(relation_file) as f:
            relation_data = RelationData.from_json(f.read())
        db.add_relation_data(relation_data)
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
