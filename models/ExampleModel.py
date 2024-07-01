import torch
import torch.nn as nn
import pytorch_lightning as pl


class DummyModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        pass
