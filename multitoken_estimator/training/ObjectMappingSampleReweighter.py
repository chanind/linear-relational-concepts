from collections import defaultdict
from typing import Sequence

import torch

from multitoken_estimator.training.ObjectMappingDataset import ObjectMappingSample


class ObjectMappingSampleReweighter:
    _counts_by_relation_and_object: dict[str, dict[str, int]]
    _num_objects_per_relation: dict[str, int]
    _reweighting_cache: dict[tuple[str, str], float]

    def __init__(self, samples: Sequence[ObjectMappingSample]):
        self._reweighting_cache = {}
        self._counts_by_relation_and_object = defaultdict(lambda: defaultdict(int))
        for sample in samples:
            self._counts_by_relation_and_object[sample.relation_name][
                sample.object_name
            ] += 1
        self._num_objects_per_relation = {
            relation_name: len(objects)
            for relation_name, objects in self._counts_by_relation_and_object.items()
        }

    def calc_samples_reweighting_tensor(
        self,
        relation_names: Sequence[str],
        object_names: Sequence[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        assert len(relation_names) == len(object_names)

        return torch.tensor(
            [
                self.calc_sample_reweighting(relation_name, object_name)
                for relation_name, object_name in zip(relation_names, object_names)
            ],
            device=device,
            dtype=dtype,
        )

    def calc_sample_reweighting(self, relation_name: str, object_name: str) -> float:
        """
        Reweight first for the object frequency in the relation, then for the
        number of objects per relation
        """
        cache_key = (relation_name, object_name)
        if cache_key not in self._reweighting_cache:
            num_samples_in_relation = sum(
                self._counts_by_relation_and_object[relation_name].values()
            )
            object_portion = (
                self._counts_by_relation_and_object[relation_name][object_name]
                / num_samples_in_relation
            )
            object_reweighting = _calc_reweighting(
                object_portion, self._num_objects_per_relation[relation_name]
            )

            num_objects_total = sum(self._num_objects_per_relation.values())
            num_relations = len(self._num_objects_per_relation)
            relation_portion = (
                self._num_objects_per_relation[relation_name] / num_objects_total
            )
            relation_reweighting = _calc_reweighting(relation_portion, num_relations)
            reweighting = object_reweighting * relation_reweighting
            self._reweighting_cache[cache_key] = reweighting
        return self._reweighting_cache[cache_key]


def _calc_reweighting(class_portion: float, num_classes: int) -> float:
    """
    Calculate reweighting to increase weight of underrepresented classes.
    Use 1 / (num_classes * class_portion) as the weighting. This is based
    on the "balanced" weighting from scikit-learn's LinearSVC class
    """
    if class_portion == 0:
        return 0
    return 1 / (num_classes * class_portion)
