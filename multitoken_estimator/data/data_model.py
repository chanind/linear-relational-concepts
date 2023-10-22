from dataclasses import dataclass
from typing import Optional

from dataclasses_json import DataClassJsonMixin


# LRE data types
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


# external types
@dataclass(frozen=True, slots=True)
class Entity(DataClassJsonMixin):
    name: str
    type: str
    alternative_names: Optional[frozenset[str]] = None


@dataclass(frozen=True, slots=True)
class RelationSample(DataClassJsonMixin):
    subject: Entity
    object: Entity


@dataclass(frozen=True, slots=True)
class RelationData(DataClassJsonMixin):
    name: str
    templates: frozenset[str]
    samples: list[RelationSample]
    zs_templates: frozenset[str] | None = None
    category: str | None = None


# data models (for storage in the DB)
@dataclass(frozen=True, slots=True)
class DataModel:
    id: int


@dataclass(frozen=True, slots=True)
class EntityTypeDataModel(DataModel):
    name: str


@dataclass(frozen=True, slots=True)
class EntityDataModel(DataModel):
    name: str
    alternative_names: Optional[frozenset[str]]
    entity_type: EntityTypeDataModel


@dataclass(frozen=True, slots=True)
class RelationDataModel(DataModel):
    name: str
    # expects a template string with a {} placeholder for the subject
    templates: frozenset[str]
    zs_templates: frozenset[str] | None = None
    category: str | None = None


@dataclass(frozen=True, slots=True)
class SampleDataModel(DataModel):
    relation: RelationDataModel
    subject: EntityDataModel
    object: EntityDataModel
