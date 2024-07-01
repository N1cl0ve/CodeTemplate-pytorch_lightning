import torch
import torch.nn as nn


class ExampleLoss(nn.Module):
    """
    An example module used for Image classification Model
    You can build your own module for your own model
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logit, labels):
        return self.loss_fn(logit, labels)
