import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset

from pyteomics.mass import fast_mass
from sklearn.model_selection import train_test_split

from .torch_helpers import RejectionSampler, zero_padding_collate, cache_path, PandasHDFDataset
from .spectrum import transform_spectrum
from .cdhit import CDHIT
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
        filter=None,
        num_workers=1,
        cache_dir=None,
        random_state=0
    ):
        super().__init__()
        self.hdf_path = hdf_path
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.cdhit_threshold = cdhit_threshold
        self.cdhit_word_length = cdhit_word_length
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.random_state = 0
        self.filter = lambda item: True if filter is None else filter

    def setup(self, stage=None):
        if self.cache_dir:
            new_path = cache_path(self.hdf_path, self.cache_dir)
        else:
            new_path = self.hdf_path

        # hmm. first double check that this still works on cluster
        self.dataset = PandasHDFDataset(
            new_path,
            primary_key='Spectrum',
            transform=transform_spectrum
        )
        
        self.sequences = self.dataset.hdf.select(
            key='Spectrum',
            columns=['sequence']
        ).iloc[:,0].tolist()
        
        cdhit = CDHIT(
            threshold=self.cdhit_threshold,
            word_length=self.cdhit_word_length
        )
        clusters = cdhit.fit_predict(list(set(self.sequences)))
        train_clusters, val_clusters = train_test_split(
            clusters, 
            train_size=self.train_val_split,
            random_state=self.random_state
        )
        train_clusters = set(train_clusters)
        val_clusters = set(val_clusters)
        self.train_sequences = [s for s, c in zip(self.sequences, clusters) if c in train_clusters]
        self.val_sequences = [s for s, c in zip(self.sequences, clusters) if c in val_clusters]
        
        self.train_dataset = RejectionSampler(
            dataset=self.dataset,
            indicator=lambda item: self.filter(item) and (item['sequence'] in self.train_sequences),
            shuffle=True
        )

        self.val_dataset = RejectionSampler(
            dataset=self.dataset,
            indicator=lambda item: self.filter(item) and (item['sequence'] in self.val_sequences),
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

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=zero_padding_collate,
            num_workers=self.num_workers,
            drop_last=False
        )
        return dataloader
    
    def predict_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=1,
#             collate_fn=zero_padding_collate,
            num_workers=1,
            drop_last=False
        )
        return dataloader