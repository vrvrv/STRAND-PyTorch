from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class emptyDataset(Dataset):

    def __init__(self):
        super().__init__()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return [0]


class emptyDataModule(LightningDataModule):
    def __init__(self):
        super(emptyDataModule, self).__init__()

    def setup(self, stage=None) -> None:
        self.train = self.test = emptyDataset()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=1,
            num_workers=1,
            shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=1,
            num_workers=1,
            shuffle=False
        )
