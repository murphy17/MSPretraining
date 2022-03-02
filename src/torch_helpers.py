import numpy as np
from numpy.random import RandomState
from copy import deepcopy
import os
import pandas as pd

bash = lambda s: os.popen(s).read().rstrip().split('\n')

from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import torch.distributed as dist

class PandasHDFDataset(Dataset):
    def __init__(self, hdf_path, primary_key, transform=None):
        self.hdf_path = hdf_path
        self.hdf = pd.HDFStore(hdf_path, mode='r')
        self.primary_key = primary_key
        self.length = self.hdf.get_storer(primary_key).nrows
        self.transform = transform
        
    def __getitem__(self, idx):
        item = {}
        item['index'] = idx
        for key in self.hdf.keys():
            value = self.hdf.select(key=key, where=f'index=={idx}')
            value = value.to_dict(orient='list')
            item[key.lstrip('/')] = value
        if self.transform is not None:
            item = self.transform(item)
        return item
    
    def __len__(self):
        return self.length
    
    def __del__(self):
        self.hdf.close()
        
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
    [cache_dir] = bash(f'echo {cache_dir}')
    bash(f'mkdir -p {cache_dir}')
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
