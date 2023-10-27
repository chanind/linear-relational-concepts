from multitoken_estimator.data.data_loaders import (
    get_relation_to_lre_type_map,
    load_lre_data,
)


def test_load_lre_data_with_limited_files() -> None:
    db = load_lre_data(only_load_files={"city_in_country.json"})
    all_relations = db.relations
    assert len(all_relations) == 1
    assert all_relations[0].name == "city in country"


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
