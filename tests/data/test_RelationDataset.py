from linear_relational.lib.util import group_items

from linear_relational_concepts.data.data_loaders import (
    LreRelation,
    load_lre_data,
    lre_relation_to_relation_and_samples,
)
from linear_relational_concepts.data.RelationDataset import RelationDataset, Sample
from linear_relational_concepts.lib.constants import DATA_DIR

DATA_FILE = DATA_DIR / "lre/factual/city_in_country.json"


def test_database_load_city_in_country_data() -> None:
    with open(DATA_FILE) as f:
        relation, samples = lre_relation_to_relation_and_samples(
            LreRelation.from_json(f.read())
        )
    db = RelationDataset()
    db.add_relation(relation)
    for sample in samples:
        db.add_sample(sample)

    assert len(db.relations) == 1
    assert len(db.get_relation_samples("city in country")) == 27
    assert len(db.get_object_names("city in country")) == 21


def test_split_full_lre_relations_dataset() -> None:
    db = load_lre_data()
    train_db, test_db = db.split()

    train_relations_where_all_objects_have_one_sample = train_db.get_relation_names()

    train_samples: list[Sample] = []
    test_samples: list[Sample] = []
    all_samples: list[Sample] = []
    for relation in db.relations:
        all_samples.extend(db.get_relation_samples(relation.name))
    for relation in train_db.relations:
        rel_samples = train_db.get_relation_samples(relation.name)
        rel_samples_by_object = group_items(rel_samples, lambda s: s.object)
        for samples in rel_samples_by_object.values():
            if (
                relation.name in train_relations_where_all_objects_have_one_sample
                and len(samples) > 1
            ):
                train_relations_where_all_objects_have_one_sample.remove(relation.name)
        train_samples.extend(rel_samples)
    for relation in test_db.relations:
        test_samples.extend(test_db.get_relation_samples(relation.name))

    train_objects = set(sample.object for sample in train_samples)
    test_objects = set(sample.object for sample in test_samples)

    assert len(train_samples) + len(test_samples) == len(all_samples)
    assert len(all_samples) == 11049
    # hardcoding these values as a form of snapshot test
    assert len(train_db.relations) == 47
    assert len(test_db.relations) == 47
    assert len(train_objects) == 3484  # 3623
    assert len(test_objects) == 426  # 465
    assert len(train_relations_where_all_objects_have_one_sample) == 14  # 11
    assert len(train_samples) == 7196  # 9250
    assert len(test_samples) == 3853  # 5320
