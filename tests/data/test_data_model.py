from multitoken_estimator.data.data_loaders import (
    LreRelation,
    lre_relation_to_relation_and_samples,
)
from multitoken_estimator.lib.constants import DATA_DIR

LRE_DATA_FILE = DATA_DIR / "lre/factual/city_in_country.json"
CUSTOM_DATA_FILE = DATA_DIR / "custom/city_in_country.json"


def test_load_city_in_country_lre_data() -> None:
    with open(LRE_DATA_FILE) as f:
        lre_relation = LreRelation.from_json(f.read())
    relation, samples = lre_relation_to_relation_and_samples(lre_relation)
    assert relation.templates == {"{} is part of", "{} is in the country of"}
    assert len(samples) == 27
    assert samples[0].subject == "New York City"
    assert samples[0].object == "United States"


# TODO: Fix this test
# def test_load_city_in_country_custom_data() -> None:
#     with open(CUSTOM_DATA_FILE) as f:
#         relation_data = RelationData.from_json(f.read())
#     assert relation_data.name == "city in country"
#     assert relation_data.templates == {"{} is located in the country of"}
#     assert len(relation_data.samples) == 3801
#     assert relation_data.samples[0].subject.name == "Toronto"
#     assert relation_data.samples[0].subject.type == "city"
#     assert relation_data.samples[0].object.name == "Canada"
#     assert relation_data.samples[0].object.type == "country"
