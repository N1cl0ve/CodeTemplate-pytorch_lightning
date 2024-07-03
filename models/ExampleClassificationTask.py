from typing import Any, Optional

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from modules.ClassificationTask.ExampleModule import ExampleConvolution, ExampleClassifier
from losses.ClassificationTask.ExampleLoss import CrossEntropyLoss
import pytorch_lightning as pl


class ExampleAlexNet(pl.LightningModule):
    """
    An example alex-net inherited from pl.LightningModule
    Use pytorch_lightning to build a general framework

    """

    def __init__(self):
        super().__init__()
        self.conv = ExampleConvolution()
        self.classifier = ExampleClassifier()
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        h = self.conv(x)
        logit = self.classifier(h)

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
        self.log('Validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('Validation Accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(list(self.conv.parameters()) +
                               list(self.classifier.parameters()),
                               lr=lr, betas=(0.5, 0.9))
        # scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[12000, 24000]),
        #     "interval": "step",
        #     "frequency": 1
        # }

        return [opt]  # , [scheduler]

