from typing import Iterable, Optional

from multitoken_estimator.data_model import Entity, RelationData, RelationSample
from multitoken_estimator.database import Database
from multitoken_estimator.PromptGenerator import (
    PromptGenerator,
    split_entry_with_dashes,
    split_entry_with_periods,
    split_entry_with_spaces,
    uppercase_entry,
)


def loc_sample(
    city: str,
    country: str,
    alternative_country_names: Optional[Iterable[str]] = None,
) -> RelationSample:
    alts = frozenset(alternative_country_names) if alternative_country_names else None
    return RelationSample(
        subject=Entity(city, "city"),
        object=Entity(country, "country", alts),
    )


def test_PromptGenerator() -> None:
    db = Database()
    db.add_relation_data(
        RelationData(
            "city in country",
            frozenset({"{subject} is located in the country of"}),
            samples=[
                loc_sample("Toronto", "Canada"),
                loc_sample("Montreal", "Canada"),
                loc_sample("Beijing", "China"),
                loc_sample("Tokyo", "Japan"),
                loc_sample("Paris", "France"),
                loc_sample("Berlin", "Germany"),
                loc_sample("Munich", "Germany"),
            ],
        )
    )

    pg = PromptGenerator(db)
    prompts = pg.generate_prompts_for_object("Canada")
    # 6x entity modifiers * 1x templates * 2x Canada objects = 14
    assert len(prompts) == 12


def test_PromptGenerator_customize_entity_modifiers() -> None:
    db = Database()
    db.add_relation_data(
        RelationData(
            "city in country",
            frozenset({"{subject} is located in the country of"}),
            samples=[
                loc_sample("Toronto", "Canada"),
                loc_sample("Beijing", "China"),
            ],
        )
    )
    pg = PromptGenerator(db)
    prompts = pg.generate_prompts_for_object(
        "Canada", entity_modifiers=[uppercase_entry]
    )
    assert len(prompts) == 1
    prompt = list(prompts)[0]
    print(prompt.text)
    assert (
        prompt.text
        == "Beijing is located in the country of CHINA\nToronto is located in the country of"
    )
    assert prompt.answer == "CANADA"


def test_split_entry_with_dashes() -> None:
    assert split_entry_with_dashes("Canada") == "C-a-n-a-d-a"
    assert split_entry_with_dashes("United States") == "U-n-i-t-e-d S-t-a-t-e-s"


def test_split_entry_with_periods() -> None:
    assert split_entry_with_periods("Canada") == "C.a.n.a.d.a"
    assert split_entry_with_periods("United States") == "U.n.i.t.e.d S.t.a.t.e.s"


def test_split_entry_with_spaces() -> None:
    assert split_entry_with_spaces("Canada") == "C a n a d a"
    assert split_entry_with_spaces("United States") == "U n i t e d S t a t e s"
