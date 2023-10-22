from typing import Callable, Optional, TypeVar

from multitoken_estimator.lib.util import stable_shuffle

from .data_model import (
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

    # select helpers

    def get_relation_names(self) -> set[str]:
        return {
            relation.name
            for relation in self.query_all(RelationDataModel)
            if relation.name is not None
        }

    def get_object_names_in_relation(self, relation_name: str) -> set[str]:
        return {
            sample.object.name
            for sample in self.query_all(SampleDataModel)
            if sample.relation.name == relation_name
        }

    # data model select

    def get_by_id(self, type: type[T], id: int | None) -> T | None:
        if id is None:
            return None
        model = self.models.get(id)
        return model if isinstance(model, type) else None

    def query_all(
        self,
        type: type[T],
        matcher: Optional[Callable[[T], bool]] = None,
        sort: bool = False,
    ) -> list[T]:
        models = [
            model
            for model in self.models.values()
            if isinstance(model, type) and (matcher is None or matcher(model))
        ]
        if sort:
            models.sort(key=lambda m: m.id)
        return models

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
        self,
        name: str,
        templates: frozenset[str],
        zs_templates: frozenset[str] | None = None,
        category: str | None = None,
    ) -> RelationDataModel:
        relation_template = self.query_one(
            RelationDataModel,
            lambda t: t.templates == templates
            and t.name == name
            and t.zs_templates == zs_templates
            and t.category == category,
        )
        if relation_template is None:
            relation_template = RelationDataModel(
                id=self._autoincrement_id(),
                name=name,
                templates=templates,
                zs_templates=zs_templates,
                category=category,
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
        self, name: str, type: str, alternative_names: Optional[frozenset[str]]
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
            relation_data.name,
            relation_data.templates,
            relation_data.zs_templates,
            relation_data.category,
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

    def _add_sample(self, sample: SampleDataModel) -> SampleDataModel:
        self.models[sample.id] = sample
        self.models[sample.subject.id] = sample.subject
        self.models[sample.object.id] = sample.object
        self.models[sample.relation.id] = sample.relation
        return sample

    def split(
        self, train_portion: float = 0.5, seed: int | str = 42
    ) -> tuple["Database", "Database"]:
        """
        Split this dataset into train / test datasets based on SampleDataModel grouped by relation
        """

        train_database = Database()
        test_database = Database()
        relations = [relation for relation in self.query_all(RelationDataModel)]
        for relation in relations:
            relation_samples = self.query_all(
                SampleDataModel, lambda s: s.relation == relation, sort=True
            )
            shuffled_samples = stable_shuffle(
                relation_samples, seed=f"{relation.name}-{seed}"
            )
            split_index = int(len(shuffled_samples) * train_portion)
            for train_sample in shuffled_samples[:split_index]:
                train_database._add_sample(train_sample)
            for test_sample in shuffled_samples[split_index:]:
                test_database._add_sample(test_sample)
        return train_database, test_database

    def reformulate_samples(
        self, modifier: Callable[[SampleDataModel], SampleDataModel]
    ) -> "Database":
        """
        Reformulate samples by a user-provided modifier fn.
        Returns a new database with the reformulated samples.
        """
        new_database = Database()
        for sample in self.query_all(SampleDataModel):
            new_sample = modifier(sample)
            new_database._add_sample(new_sample)
        return new_database
