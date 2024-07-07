from typing import Any, Optional

import torch
from torch import nn as nn

from modules.ImageClassification.AlexNetModule import Convolution, Classifier


class AlexNet(nn.Module):
    """
    An example alex-net inherited from pl.LightningModule
    Use pytorch_lightning to build a general framework

    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = Convolution(in_channels)
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        h = self.conv(x)
        logit = self.classifier(h)

        return logit
