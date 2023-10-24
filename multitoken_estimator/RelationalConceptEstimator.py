from dataclasses import dataclass

import torch

from multitoken_estimator.Concept import Concept
from multitoken_estimator.LinearRelationalEmbedding import (
    InvertedLinearRelationalEmbedding,
)
from multitoken_estimator.ObjectMappingModel import ObjectMappingModel


@dataclass
class RelationalConceptEstimator:
    inv_lre: InvertedLinearRelationalEmbedding
    object_mapping: ObjectMappingModel

    @property
    def relation(self) -> str:
        return self.inv_lre.relation

    @torch.no_grad()
    def estimate_concept(
        self,
        object_name: str,
        object_source_activation: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> Concept:
        """
        Given an object name and its source activation, return a Concept object which respresents this (r,o) pair
        """
        object_target_activation = self.object_mapping.forward(object_source_activation)
        vector = self.inv_lre.calculate_subject_activation(object_target_activation)
        return Concept(
            object=object_name,
            relation=self.inv_lre.relation,
            layer=self.inv_lre.subject_layer,
            vector=vector.to(device),
        )
