from multitoken_estimator.data_model import (
    EntityDataModel,
    EntityTypeDataModel,
    LreRelation,
    RelationDataModel,
    SampleDataModel,
    lre_relation_to_relation_data,
)
from multitoken_estimator.database import Database, get_relation_to_lre_type_map
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


def test_get_relation_to_lre_type_map() -> None:
    assert get_relation_to_lre_type_map() == {
        "adjective antonym": "linguistic",
        "adjective comparative": "linguistic",
        "adjective superlative": "linguistic",
        "characteristic gender": "bias",
        "city in country": "factual",
        "company CEO": "factual",
        "company hq": "factual",
        "country capital city": "factual",
        "country currency": "factual",
        "country language": "factual",
        "country largest city": "factual",
        "food from country": "factual",
        "fruit inside color": "commonsense",
        "fruit outside color": "commonsense",
        "landmark in country": "factual",
        "landmark on continent": "factual",
        "name birthplace": "bias",
        "name gender": "bias",
        "name religion": "bias",
        "object superclass": "commonsense",
        "occupation age": "bias",
        "occupation gender": "bias",
        "person father": "factual",
        "person lead singer of band": "factual",
        "person mother": "factual",
        "person native language": "bias",
        "person occupation": "factual",
        "person plays instrument": "factual",
        "person sport position": "factual",
        "person university": "factual",
        "plays pro sport": "factual",
        "pokemon evolution": "factual",
        "president birth year": "factual",
        "president election year": "factual",
        "product by company": "factual",
        "star constellation name": "factual",
        "substance phase of matter": "commonsense",
        "superhero archnemesis": "factual",
        "superhero person": "factual",
        "task done by tool": "commonsense",
        "task person type": "commonsense",
        "univ degree gender": "bias",
        "verb past tense": "linguistic",
        "word first letter": "linguistic",
        "word last letter": "linguistic",
        "word sentiment": "commonsense",
        "work location": "commonsense",
    }
