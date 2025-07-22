import pytorch_lightning as pl
from monai.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        predict_dataset=None,
        num_workers=8,
        batch_size=2,
        drop_last=False,
        pin_memory=True,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def setup(self, stage=None):
        if self.train_dataset is not None:
            print(f"Train samples: {len(self.train_dataset)}")
        if self.val_dataset is not None:
            print(f"Val samples: {len(self.val_dataset)}")
        if self.test_dataset is not None:
            print(f"Test samples: {len(self.test_dataset)}")
        if self.predict_dataset is not None:
            print(f"Predict samples: {len(self.predict_dataset)}")

    def train_dataloader(self):
        if self.train_dataset is None:
            return DataLoader([])

        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            prefetch_factor=1,
            persistent_workers=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return DataLoader([])

        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=1,
            pin_memory=self.pin_memory,
            shuffle=False,
            prefetch_factor=1,
            persistent_workers=True,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return DataLoader([])

        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=1,
            shuffle=False,
            prefetch_factor=1,
            persistent_workers=True,
            drop_last=self.drop_last,
        )

    def predict_dataloader(self):
        if self.predict_dataset is None:
            return DataLoader([])

        return DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=1,
            shuffle=False,
            pin_memory=self.pin_memory,
            prefetch_factor=1,
            persistent_workers=True,
            drop_last=self.drop_last,
        )
