from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin


# external types
@dataclass
class Entity(DataClassJsonMixin):
    names: set[str]
    type: str


@dataclass
class RelationSample(DataClassJsonMixin):
    subject: Entity
    object: Entity


@dataclass
class RelationData(DataClassJsonMixin):
    name: str
    templates: set[str]
    samples: list[RelationSample]


# data models (for storage in the DB)
@dataclass
class DataModel:
    id: int


@dataclass
class EntityTypeDataModel(DataModel):
    name: str


@dataclass
class EntityDataModel(DataModel):
    names: set[str]
    entity_type_id: int


@dataclass
class RelationTemplateDataModel(DataModel):
    # expects a template string with {subject} and {object} placeholders
    templates: set[str]


@dataclass
class SampleDataModel(DataModel):
    template_id: int
    subject_id: int
    object_id: int
