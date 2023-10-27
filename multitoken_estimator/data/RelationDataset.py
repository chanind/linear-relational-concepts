from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from multitoken_estimator.lib.util import stable_shuffle


@dataclass(frozen=True, slots=True)
class Relation:
    name: str
    # expects a template string with a {} placeholder for the subject
    templates: frozenset[str]
    zs_templates: frozenset[str] | None = None
    category: str | None = None


@dataclass(frozen=True, slots=True)
class Sample:
    relation: str
    subject: str
    object: str


class RelationDataset:
    samples: list[Sample]
    samples_by_relation: dict[str, list[Sample]]
    samples_by_relation_and_object: dict[str, dict[str, list[Sample]]]
    relations_by_name: dict[str, Relation]

    def __init__(self) -> None:
        self.samples = []
        self.relations_by_name = {}
        self.samples_by_relation = defaultdict(list)
        self.samples_by_relation_and_object = defaultdict(lambda: defaultdict(list))

    @property
    def relations(self) -> list[Relation]:
        return list(self.relations_by_name.values())

    def get_relation(self, relation: str) -> Relation:
        return self.relations_by_name[relation]

    def get_relation_names(self) -> set[str]:
        return set(self.relations_by_name.keys())

    def get_object_names(self, relation: str) -> set[str]:
        return set(self.samples_by_relation_and_object[relation].keys())

    def get_relation_samples(self, relation: str) -> list[Sample]:
        return self.samples_by_relation[relation]

    def get_object_samples(self, relation: str, object: str) -> list[Sample]:
        return self.samples_by_relation_and_object[relation][object]

    def add_relation(self, relation: Relation) -> None:
        self.relations_by_name[relation.name] = relation

    def add_sample(self, sample: Sample) -> None:
        self.samples.append(sample)
        self.samples_by_relation[sample.relation].append(sample)
        self.samples_by_relation_and_object[sample.relation][sample.object].append(
            sample
        )

    def split(
        self, train_portion: float = 0.5, seed: int | str = 42
    ) -> tuple["RelationDataset", "RelationDataset"]:
        """
        Split this dataset into train / test datasets based on SampleDataModel grouped by relation
        """

        train_dataset = RelationDataset()
        test_dataset = RelationDataset()
        for relation in self.relations:
            train_dataset.add_relation(relation)
            test_dataset.add_relation(relation)
            relation_samples = self.samples_by_relation[relation.name]
            shuffled_samples = stable_shuffle(
                relation_samples, seed=f"{relation.name}-{seed}"
            )
            split_index = int(len(shuffled_samples) * train_portion)
            for train_sample in shuffled_samples[:split_index]:
                train_dataset.add_sample(train_sample)
            for test_sample in shuffled_samples[split_index:]:
                test_dataset.add_sample(test_sample)
        return train_dataset, test_dataset

    def reformulate_relations(
        self, modifier: Callable[[Relation], Relation]
    ) -> "RelationDataset":
        """
        Reformulate samples by a user-provided modifier fn.
        Returns a new database with the reformulated samples.
        """
        new_dataset = RelationDataset()
        for relation in self.relations:
            new_relation = modifier(relation)
            if relation.name != new_relation.name:
                raise ValueError(
                    f"Reformulated relation name {new_relation.name} does not match original relation name {relation.name}"
                )
            new_dataset.add_relation(new_relation)
        for sample in self.samples:
            new_dataset.add_sample(sample)
        return new_dataset
