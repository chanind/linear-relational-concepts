from typing import Callable, Optional, TypeVar

from multitoken_estimator.data_model import (
    DataModel,
    EntityDataModel,
    EntityTypeDataModel,
    RelationData,
    RelationTemplateDataModel,
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

    # upsert helpers

    def get_or_create_relation_template(
        self, templates: set[str]
    ) -> RelationTemplateDataModel:
        relation_template = self.query_one(
            RelationTemplateDataModel, lambda t: t.templates == templates
        )
        if relation_template is None:
            relation_template = RelationTemplateDataModel(
                id=self._autoincrement_id(), templates=templates
            )
            self._add(relation_template)
        return relation_template

    def get_or_create_entity_type(self, name: str) -> EntityTypeDataModel:
        entity_type = self.query_one(EntityTypeDataModel, lambda t: t.name == name)
        if entity_type is None:
            entity_type = EntityTypeDataModel(id=self._autoincrement_id(), name=name)
            self._add(entity_type)
        return entity_type

    def get_or_create_entity(self, names: set[str], type: str) -> EntityDataModel:
        entity_type = self.get_or_create_entity_type(type)
        entity = self.query_one(
            EntityDataModel,
            lambda e: e.names == names and e.entity_type_id == entity_type.id,
        )
        if entity is None:
            entity = EntityDataModel(
                id=self._autoincrement_id(), names=names, entity_type_id=entity_type.id
            )
            self._add(entity)
        return entity

    # insert external data

    def add_relation_data(self, relation_data: RelationData) -> None:
        template = self.get_or_create_relation_template(relation_data.templates)
        for sample in relation_data.samples:
            subject = self.get_or_create_entity(
                sample.subject.names, sample.subject.type
            )
            object = self.get_or_create_entity(sample.object.names, sample.object.type)
            relation_model = SampleDataModel(
                id=self._autoincrement_id(),
                template_id=template.id,
                subject_id=subject.id,
                object_id=object.id,
            )
            self._add(relation_model)

    def _autoincrement_id(self) -> int:
        if len(self.models) == 0:
            return 1
        return max(self.models.keys()) + 1

    def _add(self, model: T) -> T:
        self.models[model.id] = model
        return model
