from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from multitoken_estimator.ObjectMappingModel import ObjectMappingModel
from multitoken_estimator.training.ObjectMappingDataset import ObjectMappingBatch
from multitoken_estimator.training.ObjectMappingSampleReweighter import (
    ObjectMappingSampleReweighter,
)


class ObjectMappingTrainingWrapper(pl.LightningModule):
    """Pytorch Lightning wrapper for training ObjectMappingModel."""

    object_mapping: ObjectMappingModel
    lr: float
    lr_gamma: float
    reweighter: ObjectMappingSampleReweighter | None
    training_step_outputs: list[torch.Tensor]
    validation_step_outputs: list[torch.Tensor]
    criterion: nn.Module

    def __init__(
        self,
        object_mapping: ObjectMappingModel,
        lr: float = 0.01,
        lr_gamma: float = 0.9,
        reweighter: Optional[ObjectMappingSampleReweighter] = None,
    ):
        super().__init__()
        self.object_mapping = object_mapping
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.reweighter = reweighter
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.criterion = nn.MSELoss(reduction="none")

    def configure_optimizers(self) -> tuple[list[AdamW], list[ExponentialLR]]:
        optimizer = torch.optim.AdamW(self.object_mapping.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=self.lr_gamma, verbose=True)
        return [optimizer], [scheduler]

    def training_step(
        self,
        train_batch: ObjectMappingBatch,
        _batch_idx: int,
    ) -> STEP_OUTPUT:
        loss = self._forward_step(train_batch)
        self.log("train_loss", loss)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(
        self,
        train_batch: ObjectMappingBatch,
        _batch_idx: int,
    ) -> STEP_OUTPUT:
        loss = self._forward_step(train_batch)
        self.log("validation_loss", loss)
        self.validation_step_outputs.append(loss)
        return loss

    def _forward_step(self, train_batch: ObjectMappingBatch) -> torch.Tensor:
        object_names, source_vectors, target_vectors = train_batch
        predictions = self.object_mapping(source_vectors)
        raw_loss = self.criterion(predictions, target_vectors)
        if self.reweighter is not None:
            reweighting_tensor = self.reweighter.calc_samples_reweighting_tensor(
                object_names, raw_loss.device, raw_loss.dtype
            )
            raw_loss = raw_loss * reweighting_tensor.unsqueeze(dim=1)
        loss = raw_loss.mean()
        return loss

    def on_train_epoch_end(self) -> None:
        average_training_loss = torch.mean(torch.stack(self.training_step_outputs))
        self.log("training_epoch_average", average_training_loss)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        average_validation_loss = torch.mean(torch.stack(self.validation_step_outputs))
        self.log("validation_epoch_average", average_validation_loss)
        self.validation_step_outputs.clear()
