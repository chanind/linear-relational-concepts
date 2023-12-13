from typing import Any, Sequence

import numpy as np
import torch


def train_avg_concept_vector(
    deltas: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    training_vecs = []
    for delta in deltas:
        training_vecs.append(delta.numpy())
    avg_vec = np.array(training_vecs).mean(axis=0)
    magnitude = np.linalg.norm(avg_vec)
    metadata = {
        "num_deltas": len(deltas),
    }
    return torch.FloatTensor(avg_vec / magnitude), metadata
