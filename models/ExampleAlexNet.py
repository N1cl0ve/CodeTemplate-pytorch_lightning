from modules.ExampleModule import ExampleConvolution, ExampleClassifier
from losses.ExampleLoss import ExampleLoss
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
        self.loss_fn = ExampleLoss()

    def forward(self, x):
        h = self.conv(x)
        logit = self.classifier(h)

        return logit

    def training_step(self, batch, batch_idx):
        """

        Args:
            batch:
            batch_idx:

        Returns:
            loss: training loss
        """

        inputs, labels = batch
        logit = self(inputs)
        loss = self.loss_fn(logit, labels)
        # TODO: 添加log与log_dict函数的模板与描述

        return loss


