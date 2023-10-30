from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

from multitoken_estimator.lib.util import group_items, stable_shuffle


@dataclass(frozen=True, slots=True)
class Relation:
    name: str
    # expects a template string with a {} placeholder for the subject
    templates: tuple[str, ...]
    zs_templates: tuple[str, ...] | None = None
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
        self,
        train_portion: float = 0.5,
        seed: int | str = 42,
    ) -> tuple["RelationDataset", "RelationDataset"]:
        """
        Split this dataset into train / test datasets based on SampleDataModel grouped by relation
        NOTE: the train_portion operates per relation and object. So you can get unbalanced results
        if, e.g., you have 3 samples per object, but aim for a 0.5 split, as you'd get 2 train, 1 test
        for every object.
        NOTE: this will ensure that the train set is guaranteed to have at least 1 of every object.
        this means that the train set will be larger than train_portion if there are objects
        that only appear one time in the relation, since they MUST get added to training set.
        """

        train_dataset = RelationDataset()
        test_dataset = RelationDataset()
        for relation in self.relations:
            train_dataset.add_relation(relation)
            test_dataset.add_relation(relation)
            samples = self.get_relation_samples(relation.name)
            sample_by_object = group_items(samples, lambda s: s.object)
            for object, object_samples in sample_by_object.items():
                shuffled_samples = stable_shuffle(
                    object_samples, seed=f"{relation}-{object}-{seed}"
                )
                train_samples, test_samples = _split_samples(
                    shuffled_samples, train_portion
                )
                for train_sample in train_samples:
                    train_dataset.add_sample(train_sample)
                for test_sample in test_samples:
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


def _split_samples(
    samples: Sequence[Sample], train_portion: float
) -> tuple[list[Sample], list[Sample]]:
    """
    Split samples, ensuring the first sample always goes in train.
    This method assumes the samples are already shuffled
    """
    train_samples: list[Sample] = [samples[0]]
    test_samples: list[Sample] = []
    for sample in samples[1:]:
        total_sorted = len(train_samples) + len(test_samples)
        cur_train_portion = len(train_samples) / total_sorted
        if cur_train_portion <= train_portion:
            train_samples.append(sample)
        else:
            test_samples.append(sample)
    return train_samples, test_samples
