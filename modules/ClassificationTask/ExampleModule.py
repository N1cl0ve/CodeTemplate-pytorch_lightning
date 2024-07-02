import torch
from torch import nn
from torch.nn import functional as F


class ExampleConvolution(nn.Module):
    """
    An example module used for Image classification Model
    You can build your own module for your own model
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten()
        )

    def forward(self, x):
        return self.conv(x)


class ExampleClassifier(nn.Module):
    """
    An example module used for Image classification Model
    You can build your own module for your own model
    """
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        return self.classifier(x)
