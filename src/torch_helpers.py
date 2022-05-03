import numpy as np
from numpy.random import RandomState
from copy import deepcopy
import os
import pandas as pd
from collections import defaultdict

bash = lambda s: os.popen(s).read().rstrip().split('\n')

from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import torch.distributed as dist

class NamedTensorDataset(Dataset):
    def __init__(self,name,/,**kwargs):
        self.name = name
        self.items = [dict(zip(kwargs.keys(),item)) for item in zip(*kwargs.values())]

    def __getitem__(self, idx):
        return self.items[idx]
    
    def __len__(self):
        return len(self.items)

class Group(Dataset):
    def __init__(self, dataset, groups):
        self.dataset = dataset
        self.groups = groups
        
    def __getitem__(self, idx):
        idxs = self.groups[idx]
        return [self.dataset[idx] for idx in idxs]
    
    def __len__(self):
        return len(self.groups)

# from diskcache import FanoutCache

from torch.utils.data import DataLoader
import pickle
import gzip
from tqdm import tqdm

# point of this is to wrap around datasets with slow implementations
# and just pull the whole thing into memory
class CacheDataset(Dataset):
    def __init__(
        self, 
        dataset, path, 
        *,
        transform=None,
        num_workers=1, 
        verbose=False
    ):
        self.dataset = dataset
        self.path = path
        self.num_workers = num_workers
        self.verbose = verbose
        self.transform = transform
        
        self.cache = []

        if os.path.exists(self.path):
            with gzip.open(self.path,'rb') as f:
                self.cache = pickle.load(f)
        else:
            assert not dist.is_initialized()
            dataloader = DataLoader(
                self.dataset,
                batch_size=1,
                collate_fn=lambda x: x,
                num_workers=self.num_workers,
                shuffle=False,
                drop_last=False
            )
            for batch in tqdm(dataloader,disable=not self.verbose):
                self.cache.extend(batch)
            
            with gzip.open(self.path,'wb') as f:
                pickle.dump(self.cache, f)

    def __getitem__(self, idx):
        item = self.cache[idx]
        if self.transform:
            item = self.transform(item)
        return item
    
    def __len__(self):
        return len(self.cache)
        
        
    
# class DiskCache(Dataset):
#     def __init__(self, dataset, *, shards=1, timeout=1, tmpdir=None):
#         self.dataset = dataset
#         self.timeout = timeout
#         self.shards = shards
#         self.tmpdir = tmpdir
#         self.cache = None
        
#     def __getitem__(self, idx):
#         if self.cache is None:
#             self.cache = FanoutCache(
#                 directory=self.tmpdir,
#                 shards=self.shards,
#                 timeout=self.timeout,
#                 eviction_policy='none',
#                 size_limit=int(1e10)
#             )
#         if idx not in self.cache:
#             item = self.dataset[idx]
#             self.cache[idx] = item
#         else:
#             item = self.cache[idx]
#         return item
            
#     def __len__(self):
#         return len(self.dataset)
        
#     def __del__(self):
#         if self.cache is not None:
#             self.cache.close()

# import pickle
# import lz4.frame as lz4

# class MemoryCache(Dataset):
#     def __init__(self, dataset, transform=None, compress=False):
#         self.dataset = dataset
#         self.transform = transform
#         self.compress = compress
#         self.cache = {}
    
#     def __getitem__(self, idx):
#         if idx not in self.cache:
#             item = self.dataset[idx]
#             if self.compress:
#                 self.cache[idx] = lz4.compress(pickle.dumps(item))
#             else:
#                 self.cache[idx] = item
#         item = self.cache[idx]
#         if self.compress:
#             item = pickle.loads(lz4.decompress(compressed))
#         if self.transform is not None:
#             item = self.transform(item)
#         return item
    
#     def __len__(self):
#         return len(self.dataset)
    
# from torch.utils.data import DataLoader, ConcatDataset
# from tqdm import tqdm
# class Prefetch(Dataset):
#     def __init__(
#         self,
#         dataset,
#         num_workers=0,
#         batch_size=1,
#         max_length=-1,
#         verbose=False
#     ):
#         dataloader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             collate_fn=lambda x: x,
#             num_workers=num_workers,
#             shuffle=False,
#             drop_last=False
#         )
#         self.dataset = []
#         for batch in tqdm(dataloader,disable=not verbose):
#             self.dataset.extend(batch)
#             if max_length > 0 and len(self.dataset) > max_length:
#                 self.dataset = self.dataset[:max_length]
#                 break
    
#     def __getitem__(self, idx):
#         return self.dataset[idx]
        
#     def __len__(self):
#         return len(self.dataset)
    
class RandomGroupSampler(Dataset):
    def __init__(
        self,
        dataset,
        size=1,
        replace=False,
        transform=None,
        random_state=0
    ):
        self.dataset = dataset
        self.size = size
        self.replace = replace
        self.transform = transform
        self.random_state = random_state
        self.rng = RandomState(random_state)
        
    def __getitem__(self, idx):
        items = self.dataset[idx]
        items = self.rng.choice(items,size=self.size,replace=self.replace).tolist()
        if self.transform:
            items = self.transform(items)
        return items
        
    def __len__(self):
        return len(self.dataset)
    
# class to make iteration a little more tractable
# class HDFTable:
#     def __init__(self, hdf, table, prefetch_size=1, index_key='index'):
#         self.hdf = hdf
#         self.table = table
#         self.prefetch_size = prefetch_size
#         self.index = index_key
#         self.start = -1
#         self.stop = -1

#     def __getitem__(self, idx):
#         cached = self.start <= idx < self.stop
#         if not self.prefetch_size or not cached:
#             self.start = idx
#             self.stop = idx + self.prefetch_size
#             self.df = self.hdf.select(
#                 key=self.table,
#                 where=f'({self.index}>={self.start})&({self.index}<{self.stop})'
#             )
#         if idx not in self.df.index:
#             return self.df.loc[[]]
#         return self.df.loc[[idx]]
    
class PandasHDFDataset(Dataset):
    def __init__(
        self,
        hdf_path, 
        primary_table=None,
        index_key='index',
        transform=None,
        in_memory=False,
#         lazy=False
    ):
        self.hdf_path = hdf_path
        self.index_key = index_key
        self.transform = transform
        self.primary_table = primary_table
        self.in_memory = in_memory
#         self.lazy = lazy
        
#         self.hdf = None
#         self.length = None
        
        # load whole thing into memory
        if self.in_memory:
            self.hdf = pd.HDFStore(self.hdf_path, mode='r', driver='H5FD_CORE')
        else:
            self.hdf = pd.HDFStore(self.hdf_path, mode='r')
        
        self.primary_table = self.hdf.keys()[0].lstrip('/')[0] if self.primary_table is None else self.primary_table
        self.length = self.hdf.get_storer(self.primary_table).nrows
        
#         if self.lazy:
#             self.hdf.close()
#             self.hdf = None
        
    def __getitem__(self, idx):
#         if self.lazy and self.hdf is None:
#             self.hdf = pd.HDFStore(self.hdf_path, mode='r', driver='H5FD_CORE')
        item = {}
        item['index'] = idx
        for key in self.hdf.keys():
            value = self.hdf.select(key,where=f'{self.index_key}=={idx}')
            value = value.to_dict(orient='list')
            item[key.lstrip('/')] = value
        if self.transform is not None:
            item = self.transform(item)
        return item
    
    def __len__(self):
        return self.length
    
    def __del__(self):
        if self.hdf is not None:
            self.hdf.close()
        
def group_collate(collate_fn):
    def _group_collate(grouped_items):
        batch = [collate_fn(items) for items in zip(*grouped_items)]
        return batch
    return _group_collate
        
def zero_padding_collate(items):
    if len(items) > 1:
        max_shapes = {}
        for item in items:
            for k, v in item.items():
                if np.isscalar(v): 
                    continue
                elif k not in max_shapes:
                    max_shapes[k] = np.shape(v)
                else:
                    max_shapes[k] = np.maximum(max_shapes[k],np.shape(v))
        for item in items:
            for k, max_shape in max_shapes.items():
                if len(max_shape) == 0:
                    continue
                pad = np.stack([np.zeros_like(max_shape), max_shape - np.shape(item[k])])
                pad = pad.T.astype(int)
                item[k] = np.pad(item[k], pad)
    items = default_collate(items)
    return items

def cache_path(path, cache_dir):
    cache_dir = os.path.expandvars(cache_dir)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    new_path = cache_dir + '/' + path.split('/')[-1]
    if not os.path.exists(new_path) or os.stat(new_path).st_size == 0:
        bash(f'cp -R "{path}" "{new_path}"')
    return new_path

def distribute_indices(indices):
    # per replica
    if dist.is_available() and dist.is_initialized():
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
        per_replica = len(dataset) // num_replicas
        indices = indices[rank*per_replica:(rank+1)*per_replica]

    # per worker
    worker_info = get_worker_info()
    if worker_info is not None:
        per_worker = len(indices) // worker_info.num_workers
        worker_id = worker_info.id
        indices = indices[worker_id*per_worker:(worker_id+1)*per_worker]
        
    return indices

class RejectionSampler(IterableDataset):
    def __init__(
        self, dataset, indicator, shuffle=True, random_state=0, transform=None):
        self.dataset = dataset
        self.indicator = deepcopy(indicator)
        self.shuffle = shuffle
        self.random_state = random_state
        self.transform = transform
        self.rng = RandomState(random_state)
        self.indices = None

    def __iter__(self):
        if self.indices is None:
            indices = np.arange(len(self.dataset))
            if self.shuffle:
                self.rng.shuffle(indices)
            self.indices = distribute_indices(indices)
            
        for idx in self.indices:
            item = self.dataset[idx]
            if self.indicator(item):
                if self.transform:
                    item = self.transform(item)
                yield item

def start_tensorboard(login_node, tmux_name='tensorboard', logging_dir=None):
    if logging_dir is None:
        logging_dir = os.getcwd() + '/lightning_logs'
    logging_dir = logging_dir
    script_path = os.path.dirname(os.path.abspath(__file__)) + '/tensorboard.sh'

    bash(f'chmod +x {script_path}')
    bash(f'ssh {login_node} \'tmux kill-session -t {tmux_name}; tmux new-session -s {tmux_name} -d srun --resv-ports=1 --pty bash -i -c "{script_path} {logging_dir}"\'')

from tqdm import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar
class NoValProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar