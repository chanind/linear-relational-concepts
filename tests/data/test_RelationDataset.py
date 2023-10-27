from multitoken_estimator.data.data_loaders import (
    LreRelation,
    lre_relation_to_relation_and_samples,
)
from multitoken_estimator.data.RelationDataset import RelationDataset
from multitoken_estimator.lib.constants import DATA_DIR

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
