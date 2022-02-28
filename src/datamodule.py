import numpy as np
from pyteomics.mass import fast_mass

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .torch_helpers import RejectionSampler, zero_padding_collate, cache_path, PandasHDFDataset
from .spectrum import transform_spectrum
from .constants import MSConstants
C = MSConstants()

class MSDataModule(LightningDataModule):
    def __init__(
        self,
        hdf_path,
        batch_size,
        train_filter,
        valid_filter,
        num_workers=1,
        cache_dir=None
    ):
        super().__init__()
        self.hdf_path = hdf_path
        self.batch_size = batch_size
        self.train_filter = train_filter
        self.valid_filter = valid_filter
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.primary_key = 'Spectrum'

    def setup(self, stage=None):
        if self.cache_dir:
            new_path = cache_path(self.hdf_path, self.cache_dir)
        else:
            new_path = self.hdf_path

        # hmm. first double check that this still works on cluster
        self.dataset = PandasHDFDataset(
            new_path,
            primary_key=self.primary_key,
            transform=transform_spectrum,
        )

        self.train_dataset = RejectionSampler(
            dataset=self.dataset,
            indicator=self.train_filter,
            shuffle=True
        )

        self.valid_dataset = RejectionSampler(
            dataset=self.dataset,
            indicator=self.valid_filter,
            shuffle=False
        )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=zero_padding_collate,
            num_workers=self.num_workers,
            drop_last=True
        )
        return dataloader

    def valid_dataloader(self):
        dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            collate_fn=zero_padding_collate,
            num_workers=self.num_workers,
            drop_last=False
        )
        return dataloader