import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import pytorch_lightning as pl


class MNISTDataset(pl.LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 batch_size: list,
                 train_val_split: float):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size if len(batch_size) == 3 else batch_size + [1]
        self.train_val_split = [train_val_split, 1. - train_val_split]

        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

    def prepare_data(self) -> None:
        # Download data
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:

        if stage == 'fit':
            mnist = datasets.MNIST(self.data_dir, train=True, transform=self.train_transform, download=False)
            self.train_dataset, self.val_dataset = random_split(mnist, self.train_val_split)

        if stage == 'test':
            self.test_dataset = datasets.MNIST(self.data_dir, train=False, transform=self.test_transform, download=False)

    def train_dataloader(self) -> TRAIN_DATALOADERS:

        return DataLoader(self.train_dataset, batch_size=self.batch_size[0], shuffle=True, pin_memory=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:

        return DataLoader(self.val_dataset, batch_size=self.batch_size[1], shuffle=False, pin_memory=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:

        return DataLoader(self.test_dataset, batch_size=self.batch_size[2], shuffle=False, pin_memory=True)

