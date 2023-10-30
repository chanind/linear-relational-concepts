from collections import defaultdict

from multitoken_estimator.data.data_loaders import (
    LreRelation,
    load_lre_data,
    lre_relation_to_relation_and_samples,
)
from multitoken_estimator.lib.constants import DATA_DIR


def test_load_lre_data_with_limited_files() -> None:
    db = load_lre_data(only_load_files={"city_in_country.json"})
    all_relations = db.relations
    assert len(all_relations) == 1
    assert all_relations[0].name == "city in country"


LRE_DATA_FILE = DATA_DIR / "lre/factual/city_in_country.json"


def test_load_city_in_country_lre_data() -> None:
    with open(LRE_DATA_FILE) as f:
        lre_relation = LreRelation.from_json(f.read())
    relation, samples = lre_relation_to_relation_and_samples(lre_relation)
    assert relation.templates == {"{} is part of", "{} is in the country of"}
    assert len(samples) == 27
    assert samples[0].subject == "New York City"
    assert samples[0].object == "United States"


def test_load_lre_data() -> None:
    dataset = load_lre_data()
    assert len(dataset.relations) == 47
    relations_by_category = defaultdict(list)
    lre_samples_by_category = defaultdict(list)
    for relation in dataset.relations:
        relations_by_category[relation.category].append(relation)
        samples = dataset.get_relation_samples(relation.name)
        lre_samples_by_category[relation.category].extend(samples)
    assert set(relations_by_category.keys()) == {
        "commonsense",
        "bias",
        "factual",
        "linguistic",
    }
    # These stats match what's in the paper
    assert len(relations_by_category["commonsense"]) == 8
    assert len(relations_by_category["bias"]) == 7
    assert len(relations_by_category["factual"]) == 26
    assert len(relations_by_category["linguistic"]) == 6
    assert len(lre_samples_by_category["commonsense"]) == 369
    assert len(lre_samples_by_category["bias"]) == 212
    assert len(lre_samples_by_category["factual"]) == 9675
    assert len(lre_samples_by_category["linguistic"]) == 793


def test_load_lre_dataset_fixes_messed_up_unicode() -> None:
    dataset = load_lre_data()
    for relation in dataset.relations:
        assert "\\" not in relation.name
        assert "\\" not in relation.category
        for template in relation.templates:
            assert "\\" not in template
        for template in relation.zs_templates or []:
            assert "\\" not in template
        for sample in dataset.get_relation_samples(relation.name):
            assert "\\" not in sample.subject
            assert "\\" not in sample.object
