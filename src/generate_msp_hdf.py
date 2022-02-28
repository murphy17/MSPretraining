import pandas as pd
import json, re, os, sys
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
import time

bash = lambda s: os.popen(s).read().rstrip().split('\n')

class MSPParser:
    def __init__(
        self,
        hdf_path,
        schema_path,
        metadata_path=None,
        cache_dir='.', 
        num_workers=1,
        chunk_size=100,
        progress=True,
        limit=0
    ):
        self.hdf_path = hdf_path
        self.progress = progress
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.limit = limit
        
        if metadata_path:
            self.metadata = pd.read_csv(metadata_path, index_col=0).to_dict(orient='index')
        else:
            self.metadata = None
            
        with open(schema_path,'r') as f:
            self.schema = json.load(f)
            
        [self.cache_dir] = bash(f'echo {cache_dir}')
        bash(f'mkdir -p {self.cache_dir}')
        
        self.hdf_args = dict(format='table', append=True, min_itemsize=300, complevel=9, complib='blosc:lz4')
        self.re_match = re.compile(r'"([^\d]+)(\d+)(-[A-Z0-9]+)?(?:\^(\d+))?\/(-?\d+(?:\.\d+)?)ppm"')
        
        self.hdf = None
        
    def parse(self, *msp_paths):
        items = []
        for msp_path in msp_paths:
            if self.progress:
                print(f'Splitting {msp_path.split("/")[-1]}', file=sys.stderr)
            items += self._split_msp(msp_path)

        hdf = pd.HDFStore(self.hdf_path, mode='w')    
    
        print(f'Parsing {len(items)} spectra', file=sys.stderr)
        
        num_processed = 0
        
        with Parallel(self.num_workers) as pool:
            if self.progress:
                pbar = tqdm()
            for i in range(0, len(items), self.chunk_size):
                buffer = pool(
                    delayed(self._parse_split)(path, name, index + num_processed)
                    for index, (path, name) in enumerate(items[i:i+self.chunk_size])
                )
                self._append_hdf(hdf, buffer)
                num_processed += len(buffer)
                if self.progress:
                    pbar.update(len(buffer))
        
        print()
        print(f'Cleaning up', file=sys.stderr)
        
        hdf.close()
        
        for msp_path in msp_paths:
            self._cleanup_msp(msp_path)
        
        # bash(f'cp {self.hdf_path} {hdf_path}')
        
        return num_processed
    
    def _append_hdf(self, hdf, list_of_dfs):
        dfs = defaultdict(list)
        for item in list_of_dfs:
            for key, df in item.items():
                dfs[key].append(df)
        for key in dfs:
            df = pd.concat(dfs[key])
            hdf.put(key=key, value=df, **self.hdf_args)
    
    def _split_msp(self, path):
        bash(f'cp -n {path} {self.cache_dir}')
        name, ext = path.split('/')[-1].split('.')
        if self.limit > 0:
            n = self.limit
        else:
            n = '*'
        csplit_cmd = f"csplit --suppress-matched --elide-empty-files --prefix '{name}' --suffix-format='_%d.{ext}' '{name}.{ext}' '/^\s*$/' '{{{n}}}'"
        if self.progress:
            csplit_cmd += " | tqdm"
        bash(f'cd {self.cache_dir} && {csplit_cmd}')
        find_cmd = f'find {self.cache_dir} -type f -name "{name}_*.{ext}"'
        paths = bash(find_cmd)
        parents = [name + '.' + ext] * len(paths)
        return list(zip(paths, parents))
    
    def _cleanup_msp(self, path):
        name, ext = path.split('/')[-1].split('.')
        find_cmd = f'find {self.cache_dir} -type f -name "{name}_*.{ext}" -delete'
        bash(find_cmd)
    
    def _parse_split(self, path, name, index):
        with open(path,'r') as f:
            header = {}
            fragments = []
            for line in f:
                if ':' in line:
                    key, val = re.split(r': ',line.strip(),maxsplit=1)
                    key = key.lower().replace(' ','_')
                    header[key] = val

                elif line[0].isdigit():
                    mz, intensity, match = line.strip().split('\t')
                    [(ion, length, loss, charge, ppm)] = re.findall(self.re_match, match)
                    charge = '1' if len(charge) == 0 else charge
                    fragments.append((float(mz), float(intensity), ion, int(length), loss, int(charge), float(ppm)))

            spectrum = {}
            spectrum['sequence'], charge = header['name'].split('/')
            spectrum['charge'] = int(charge)
            
            if self.metadata:
                spectrum.update(self.metadata[name])

            comment = {key.lower(): val for key, val in re.findall(r'([^=]+)=([^ ]+) ?', header['comment'])}
            spectrum['irt'] = float('nan') if comment['irt'] == 'NA' else float(comment['irt'])

            mods = []
            for m in comment['mods'].split('/')[1:]:
                pos, aa, mod = m.split(',')
                mods.append((int(pos), aa, mod))

            dfs = {
                'Spectrum': pd.DataFrame([spectrum], columns=self.schema['Spectrum']),
                'Modification': pd.DataFrame(mods, columns=self.schema['Modification']),
                'Fragment': pd.DataFrame(fragments, columns=self.schema['Fragment']),
            }

            for key, df in dfs.items():
                df.index = [index] * len(df)
                dfs[key] = df.astype(self.schema[key])

            return dfs
        
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--msps', type=str, nargs='+')
    parser.add_argument('--hdf', type=str)
    parser.add_argument('--schema', type=str)
    parser.add_argument('--metadata', type=str, default=None)
    parser.add_argument('--cache-dir', type=str, default='$TMPDIR')
    parser.add_argument('--progress', type=bool, default=True)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    
    msp = MSPParser(
        hdf_path=args.hdf, 
        schema_path=args.schema,
        metadata_path=args.metadata,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
        progress=args.progress,
        chunk_size=args.chunk_size,
        limit=args.limit
    )
    
    msp.parse(*args.msps)