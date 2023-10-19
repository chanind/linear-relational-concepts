from multitoken_estimator.data_model import (
    EntityDataModel,
    EntityTypeDataModel,
    LreRelation,
    RelationDataModel,
    SampleDataModel,
    lre_relation_to_relation_data,
)
from multitoken_estimator.database import Database
from multitoken_estimator.lib.constants import DATA_DIR

DATA_FILE = DATA_DIR / "lre/factual/city_in_country.json"


def test_database_load_city_in_country_data() -> None:
    with open(DATA_FILE) as f:
        relation_data = lre_relation_to_relation_data(LreRelation.from_json(f.read()))
    db = Database()
    db.add_relation_data(relation_data)

    entity_types = db.query_all(EntityTypeDataModel)
    entities = db.query_all(EntityDataModel)
    relation_templates = db.query_all(RelationDataModel)
    samples = db.query_all(SampleDataModel)
    assert len(entity_types) == 2
    assert {e.name for e in entity_types} == {"city", "country"}
    assert len(entities) == 48
    assert len(samples) == 27
    assert len(relation_templates) == 1
