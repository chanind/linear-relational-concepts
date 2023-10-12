from multitoken_estimator.constants import DATA_DIR
from multitoken_estimator.data_model import (
    EntityDataModel,
    EntityTypeDataModel,
    RelationData,
    RelationDataModel,
    SampleDataModel,
)
from multitoken_estimator.database import Database

DATA_FILE = DATA_DIR / "city_in_country.json"


def test_database_load_city_in_country_data() -> None:
    with open(DATA_FILE) as f:
        relation_data = RelationData.from_json(f.read())
    db = Database()
    db.add_relation_data(relation_data)

    entity_types = db.query_all(EntityTypeDataModel)
    entities = db.query_all(EntityDataModel)
    relation_templates = db.query_all(RelationDataModel)
    samples = db.query_all(SampleDataModel)
    assert len(entity_types) == 2
    assert {e.name for e in entity_types} == {"city", "country"}
    assert len(entities) == 3915
    assert len(samples) == 3801
    assert len(relation_templates) == 1
