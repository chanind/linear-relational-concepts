from multitoken_estimator.constants import DATA_DIR
from multitoken_estimator.data_model import RelationData

DATA_FILE = DATA_DIR / "city_in_country.json"


def test_load_city_in_country_relation_data() -> None:
    with open(DATA_FILE) as f:
        relation_data = RelationData.from_json(f.read())
    assert relation_data.name == "city in country"
    assert relation_data.templates == {"{subject} is located in the country of"}
    assert len(relation_data.samples) == 3801
    assert relation_data.samples[0].subject.name == "Toronto"
    assert relation_data.samples[0].subject.type == "city"
    assert relation_data.samples[0].object.name == "Canada"
    assert relation_data.samples[0].object.type == "country"
