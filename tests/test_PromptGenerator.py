from multitoken_estimator.data.RelationDataset import Relation, RelationDataset, Sample
from multitoken_estimator.PromptGenerator import (
    AUGMENTATION_MODIFIERS,
    PromptGenerator,
    split_with_dashes_modifier,
    split_with_dashes_uppercase_modifier,
    split_with_periods_modifier,
    split_with_spaces_modifier,
    uppercase_modifier,
)


def loc_sample(
    city: str,
    country: str,
) -> Sample:
    return Sample(
        relation="city in country",
        subject=city,
        object=country,
    )


def test_PromptGenerator() -> None:
    db = RelationDataset()
    db.add_relation(
        Relation("city in country", templates=("{} is located in the country of",))
    )

    db.add_sample(loc_sample("Toronto", "Canada"))
    db.add_sample(loc_sample("Montreal", "Canada"))
    db.add_sample(loc_sample("Beijing", "China"))
    db.add_sample(loc_sample("Tokyo", "Japan"))
    db.add_sample(loc_sample("Paris", "France"))
    db.add_sample(loc_sample("Berlin", "Germany"))
    db.add_sample(loc_sample("Munich", "Germany"))

    pg = PromptGenerator(db)
    prompts = pg.generate_prompts_for_object(
        relation_name="city in country",
        object_name="Canada",
        entity_modifiers=AUGMENTATION_MODIFIERS,
    )
    # 6x entity modifiers * 1x templates * 2x Canada objects = 12
    assert len(prompts) == 12


def test_PromptGenerator_customize_entity_modifiers() -> None:
    db = RelationDataset()
    db.add_relation(
        Relation("city in country", templates=("{} is located in the country of",))
    )

    db.add_sample(loc_sample("Toronto", "Canada"))
    db.add_sample(loc_sample("Beijing", "China"))
    pg = PromptGenerator(db)
    prompts = pg.generate_prompts_for_object(
        relation_name="city in country",
        object_name="Canada",
        entity_modifiers=[uppercase_modifier],
    )
    assert len(prompts) == 1
    prompt = list(prompts)[0]
    assert (
        prompt.text
        == "Beijing is located in the country of CHINA\nToronto is located in the country of"
    )
    assert prompt.answer == "CANADA"


def test_PromptGenerator_generate_prompts_for_relation() -> None:
    db = RelationDataset()
    db.add_relation(
        Relation("city in country", templates=("{} is located in the country of",))
    )
    db.add_sample(loc_sample("Toronto", "Canada"))
    db.add_sample(loc_sample("Beijing", "China"))

    pg = PromptGenerator(db)
    prompts = pg.generate_prompts_for_relation("city in country", entity_modifiers=None)
    assert len(prompts) == 2


def test_split_with_dashes_modifier() -> None:
    assert split_with_dashes_modifier("Canada") == "C-a-n-a-d-a"
    assert split_with_dashes_modifier("United States") == "U-n-i-t-e-d S-t-a-t-e-s"


def test_split_with_periods_modifier() -> None:
    assert split_with_periods_modifier("Canada") == "C.a.n.a.d.a"
    assert split_with_periods_modifier("United States") == "U.n.i.t.e.d S.t.a.t.e.s"


def test_split_with_spaces_modifier() -> None:
    assert split_with_spaces_modifier("Canada") == "C a n a d a"
    assert split_with_spaces_modifier("United States") == "U n i t e d S t a t e s"


def test_split_with_dashes_uppercase_modifier() -> None:
    assert split_with_dashes_uppercase_modifier("Canada") == "C-A-N-A-D-A"
    assert (
        split_with_dashes_uppercase_modifier("United States")
        == "U-N-I-T-E-D S-T-A-T-E-S"
    )
