from typing import Any, Optional

import torch
from torch import nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT

from utils import instantiate_from_config
import pytorch_lightning as pl


class ImageClassificationTask(pl.LightningModule):
    """
    An example alex-net inherited from pl.LightningModule
    Use pytorch_lightning to build a general framework

    """

    def __init__(self, *, model_config, loss_config):
        super().__init__()
        self.model: nn.Module = instantiate_from_config(model_config)
        self.loss_fn: nn.Module = instantiate_from_config(loss_config)

    def forward(self, x):
        logit = self.model(x)

        return logit

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.

        Args:
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (``int``): Integer displaying index of this batch

        Return:
            Any of.

            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``
            - ``None`` - Training will skip to the next batch.
        """
        inputs, labels = batch
        logit = self(inputs)
        loss = self.loss_fn(logit, labels)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        """
        It is recommended to validate on single device to
        ensure each sample/batch gets evaluated exactly once.

        Args:
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (``int``): Integer displaying index of this batch

        Returns:
            None
        """
        inputs, labels = batch
        logit = self(inputs)
        loss = self.loss_fn(logit, labels)
        accuracy = torch.sum((labels == torch.argmax(logit, dim=1))).item() / (len(labels) * 1.0)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_Accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.9))
        # scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[12000, 24000]),
        #     "interval": "step",
        #     "frequency": 1
        # }

        return [opt]  # , [scheduler]

