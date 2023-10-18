import argparse
from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin

from multitoken_estimator.constants import DATA_DIR
from multitoken_estimator.data_model import Entity, RelationData, RelationSample


@dataclass
class LreProperties(DataClassJsonMixin):
    relation_type: str
    domain_name: str
    range_name: str
    symmetric: bool


@dataclass
class LreSample(DataClassJsonMixin):
    subject: str
    object: str


@dataclass
class LreRelation(DataClassJsonMixin):
    name: str
    prompt_templates: list[str]
    prompt_templates_zs: list[str]
    properties: LreProperties
    samples: list[LreSample]


def main(lre_file: str) -> None:
    with open(lre_file, "r") as f:
        relation = LreRelation.from_json(f.read())
    relation_data = RelationData(
        name=relation.name,
        templates=frozenset(relation.prompt_templates),
        samples=[
            RelationSample(
                subject=Entity(
                    name=sample.subject, type=relation.properties.domain_name
                ),
                object=Entity(name=sample.object, type=relation.properties.range_name),
            )
            for sample in relation.samples
        ],
    )
    target_file_name = relation.name.replace(" ", "_") + ".json"
    with open(DATA_DIR / target_file_name, "w") as f:
        f.write(relation_data.to_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lre-file", required=True)
    args = parser.parse_args()
    main(lre_file=args.lre_file)
