from multitoken_estimator.data_model import (
    LreRelation,
    RelationData,
    lre_relation_to_relation_data,
)
from multitoken_estimator.lib.constants import DATA_DIR

LRE_DATA_FILE = DATA_DIR / "lre/factual/city_in_country.json"
CUSTOM_DATA_FILE = DATA_DIR / "custom/city_in_country.json"


def test_load_city_in_country_lre_data() -> None:
    with open(LRE_DATA_FILE) as f:
        lre_relation = LreRelation.from_json(f.read())
    relation_data = lre_relation_to_relation_data(lre_relation)
    assert relation_data.templates == {"{} is part of", "{} is in the country of"}
    assert len(relation_data.samples) == 27
    assert relation_data.samples[0].subject.name == "New York City"
    assert relation_data.samples[0].subject.type == "city"
    assert relation_data.samples[0].object.name == "United States"
    assert relation_data.samples[0].object.type == "country"


def test_load_city_in_country_custom_data() -> None:
    with open(CUSTOM_DATA_FILE) as f:
        relation_data = RelationData.from_json(f.read())
    assert relation_data.name == "city in country"
    assert relation_data.templates == {"{} is located in the country of"}
    assert len(relation_data.samples) == 3801
    assert relation_data.samples[0].subject.name == "Toronto"
    assert relation_data.samples[0].subject.type == "city"
    assert relation_data.samples[0].object.name == "Canada"
    assert relation_data.samples[0].object.type == "country"
