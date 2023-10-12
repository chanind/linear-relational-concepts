from typing import Callable, Optional, TypeVar

from multitoken_estimator.constants import DATA_DIR
from multitoken_estimator.data_model import (
    DataModel,
    EntityDataModel,
    EntityTypeDataModel,
    RelationData,
    RelationDataModel,
    SampleDataModel,
)

T = TypeVar("T", bound=DataModel)


class Database:
    models: dict[int, DataModel]

    def __init__(self) -> None:
        self.models = {}

    # data model select

    def get_by_id(self, type: type[T], id: int | None) -> T | None:
        if id is None:
            return None
        model = self.models.get(id)
        return model if isinstance(model, type) else None

    def query_all(
        self, type: type[T], matcher: Optional[Callable[[T], bool]] = None
    ) -> list[T]:
        return [
            model
            for model in self.models.values()
            if isinstance(model, type) and (matcher is None or matcher(model))
        ]

    def query_one(
        self, type: type[T], matcher: Optional[Callable[[T], bool]] = None
    ) -> T | None:
        models = self.query_all(type, matcher)
        return models[0] if len(models) > 0 else None

    def query_one_or_throw(
        self, type: type[T], matcher: Optional[Callable[[T], bool]] = None
    ) -> T:
        model = self.query_one(type, matcher)
        if model is None:
            raise ValueError(f"No {type.__name__} found matching query")
        return model

    # upsert helpers

    def get_or_create_relation(
        self, name: str, templates: set[str]
    ) -> RelationDataModel:
        relation_template = self.query_one(
            RelationDataModel, lambda t: t.templates == templates
        )
        if relation_template is None:
            relation_template = RelationDataModel(
                id=self._autoincrement_id(), name=name, templates=templates
            )
            self._add(relation_template)
        return relation_template

    def get_or_create_entity_type(self, name: str) -> EntityTypeDataModel:
        entity_type = self.query_one(EntityTypeDataModel, lambda t: t.name == name)
        if entity_type is None:
            entity_type = EntityTypeDataModel(id=self._autoincrement_id(), name=name)
            self._add(entity_type)
        return entity_type

    def get_or_create_entity(
        self, name: str, type: str, alternative_names: Optional[set[str]]
    ) -> EntityDataModel:
        entity_type = self.get_or_create_entity_type(type)
        entity = self.query_one(
            EntityDataModel,
            lambda e: e.name == name
            and e.entity_type == entity_type
            and e.alternative_names == alternative_names,
        )
        if entity is None:
            entity = EntityDataModel(
                id=self._autoincrement_id(),
                name=name,
                alternative_names=alternative_names,
                entity_type=entity_type,
            )
            self._add(entity)
        return entity

    # insert external data

    def add_relation_data(self, relation_data: RelationData) -> None:
        relation = self.get_or_create_relation(
            relation_data.name, relation_data.templates
        )
        for sample in relation_data.samples:
            subject = self.get_or_create_entity(
                sample.subject.name,
                sample.subject.type,
                sample.subject.alternative_names,
            )
            object = self.get_or_create_entity(
                sample.object.name, sample.object.type, sample.object.alternative_names
            )
            relation_model = SampleDataModel(
                id=self._autoincrement_id(),
                relation=relation,
                subject=subject,
                object=object,
            )
            self._add(relation_model)

    def _autoincrement_id(self) -> int:
        if len(self.models) == 0:
            return 1
        return max(self.models.keys()) + 1

    def _add(self, model: T) -> T:
        self.models[model.id] = model
        return model


def load_all_data() -> Database:
    db = Database()
    for relation_file in DATA_DIR.glob("*.json"):
        with open(relation_file) as f:
            relation_data = RelationData.from_json(f.read())
        db.add_relation_data(relation_data)
    return db
