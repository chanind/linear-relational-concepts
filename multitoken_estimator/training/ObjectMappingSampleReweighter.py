from collections import defaultdict
from typing import Sequence

import torch

from multitoken_estimator.training.ObjectMappingDataset import ObjectMappingSample


class ObjectMappingSampleReweighter:
    _object_portions: dict[str, float]
    _total_objects: int

    def __init__(self, samples: Sequence[ObjectMappingSample]):
        counts_by_object: dict[str, int] = defaultdict(int)
        for sample in samples:
            counts_by_object[sample.object_name] += 1
        self._total_objects = sum(counts_by_object.values())
        self._object_portions = {
            object_name: count / self._total_objects
            for object_name, count in counts_by_object.items()
        }

    def calc_samples_reweighting_tensor(
        self,
        object_names: Sequence[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.tensor(
            [self.calc_sample_reweighting(object_name) for object_name in object_names],
            device=device,
            dtype=dtype,
        )

    def calc_sample_reweighting(self, object_name: str) -> float:
        """
        Reweight first for the object frequency in the relation, then for the
        number of objects per relation
        """
        return _calc_reweighting(
            self._object_portions[object_name], self._total_objects
        )


def _calc_reweighting(class_portion: float, num_classes: int) -> float:
    """
    Calculate reweighting to increase weight of underrepresented classes.
    Use 1 / (num_classes * class_portion) as the weighting. This is based
    on the "balanced" weighting from scikit-learn's LinearSVC class
    """
    if class_portion == 0:
        return 0
    return 1 / (num_classes * class_portion)
