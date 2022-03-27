import numpy as np
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset

from pyteomics.mass import fast_mass
from sklearn.model_selection import train_test_split

from .torch_helpers import RejectionSampler, zero_padding_collate, cache_path, PandasHDFDataset
from .spectrum import transform_spectrum
from .cdhit import cdhit_split
from .constants import MSConstants
C = MSConstants()

class MSDataModule(LightningDataModule):
    def __init__(
        self,
        hdf_path,
        batch_size,
        train_val_split,
        cdhit_threshold,
        cdhit_word_length,
        tmp_env=None,
        num_workers=1,
        random_state=0,
        **kwargs
    ):
        super().__init__()
        self.hdf_path = hdf_path
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.cdhit_threshold = cdhit_threshold
        self.cdhit_word_length = cdhit_word_length
        self.num_workers = num_workers
        self.tmp_env = tmp_env
        self.random_state = 0
        # disabled rejection sampling for now
#         self.filter = lambda item: True if filter is None else filter

    def setup(self, stage=None):
        if self.tmp_env:
            self.hdf_path = cache_path(self.hdf_path, os.environ[self.tmp_env])

        # hmm. first double check that this still works on cluster
        self.dataset = PandasHDFDataset(
            self.hdf_path,
            primary_key='Spectrum',
            transform=transform_spectrum
        )
        
        self.sequences = self.dataset.hdf.select(
            key='Spectrum',
            columns=['sequence']
        ).iloc[:,0].tolist()
        
        train_seqs, val_seqs, train_idxs, val_idxs = cdhit_split(
            self.sequences,
            range(len(self.sequences)),
            split=self.train_val_split,
            threshold=self.cdhit_threshold,
            word_length=self.cdhit_word_length,
            random_state=self.random_state
        )
        self.train_dataset = Subset(self.dataset, train_idxs)
        self.val_dataset = Subset(self.dataset, val_idxs)

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=zero_padding_collate,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=zero_padding_collate,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
        return dataloader
    
    def predict_dataloader(self, shuffle=False):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=zero_padding_collate,
            num_workers=1,
            shuffle=shuffle,
            drop_last=False
        )
        return dataloader