import random

import numpy as np
import numpy.typing as npt
import torch
from sklearn.svm import LinearSVC


def train_concept_vector_via_svm(
    pos_activations: list[torch.Tensor],
    neg_activations: list[torch.Tensor],
    max_iter: int = 1000,
    class_weight: str = "balanced",
    fit_intercept: bool = False,
) -> torch.Tensor:
    """
    Train a concept via SVM on the given positive and negative activations
    """
    svm = LinearSVC(
        max_iter=max_iter, class_weight=class_weight, fit_intercept=fit_intercept
    )
    pos_labels = np.ones(len(pos_activations))
    neg_labels = np.zeros(len(neg_activations))
    pos_activations_tensor = torch.stack(pos_activations)
    neg_activations_tensor = torch.stack(neg_activations)
    all_activations = (
        torch.cat([pos_activations_tensor, neg_activations_tensor])
        .type(torch.float32)
        .cpu()
        .numpy()
    )
    all_labels = np.concatenate([pos_labels, neg_labels])
    x_train, y_train = _shuffle(all_activations, all_labels)
    svm.fit(x_train, y_train)
    vec = torch.FloatTensor(svm.coef_[0])
    return vec / vec.norm()


def _shuffle(
    x_list: npt.NDArray[np.float64], y_list: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    z_list = list(zip(x_list, y_list))
    random.shuffle(z_list)
    x_shuffled, y_shuffled = zip(*z_list)
    return np.stack(x_shuffled), np.stack(y_shuffled)
