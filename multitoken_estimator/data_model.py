from dataclasses import dataclass
from typing import Optional

from dataclasses_json import DataClassJsonMixin


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
    # expects a template string with {subject} and {object} placeholders
    templates: frozenset[str]


@dataclass(frozen=True, slots=True)
class SampleDataModel(DataModel):
    relation: RelationDataModel
    subject: EntityDataModel
    object: EntityDataModel
