import numpy as np
import pandas as pd
import re
import json

from tqdm import tqdm as _tqdm
def tqdm(*args, **kwargs):
    return _tqdm(*args, leave=False, position=0, **kwargs)

def build_msp_hdf(msp_paths, hdf_path, schema_path, metadata_path=None, progress=False, limit_per_file=0):
    with open(schema_path,'r') as f:
        schema = json.load(f)

    if metadata_path:
        metadata_df = pd.read_csv(metadata_path,index_col=0)

    if progress:
        pbar = tqdm()
    with pd.HDFStore(hdf_path, mode='w') as hdf:
        index = 0
        for msp_path in msp_paths:
            n = 0
            if metadata_path:
                metadata = metadata_df.loc[msp_path.split('/')[-1]].to_dict()
            else:
                metadata = {}
            with open(msp_path,'r') as f:
                header = {}
                fragments = []
                for line in f:
                    if ':' in line:
                        key, val = re.split(r': ',line.strip(),maxsplit=1)
                        key = key.lower().replace(' ','_')
                        header[key] = val

                    elif line[0].isdigit():
                        mz, intensity, match = line.strip().split('\t')
                        [(ion, length, loss, charge, ppm)] = re.findall(r'"([^\d]+)(\d+)(-[A-Z0-9]+)?(?:\^(\d+))?\/(-?\d+(?:\.\d+)?)ppm"', match)
                        charge = '1' if len(charge) == 0 else charge
                        fragments.append((float(mz), float(intensity), ion, int(length), loss, int(charge), float(ppm)))

                    elif line == '\n':
                        spectrum = {}
                        spectrum['sequence'], charge = header['name'].split('/')
                        spectrum['charge'] = int(charge)

                        spectrum.update(metadata)

                        comment = {key.lower(): val for key, val in re.findall(r'([^=]+)=([^ ]+) ?', header['comment'])}
                        spectrum['irt'] = float('nan') if comment['irt'] == 'NA' else float(comment['irt'])

                        mods = []
                        for m in comment['mods'].split('/')[1:]:
                            pos, aa, mod = m.split(',')
                            mods.append((int(pos), aa, mod))

                        dfs = {
                            'Spectrum': pd.DataFrame([spectrum]),
                            'Modification': pd.DataFrame(mods, columns=range(3)),
                            'Fragment': pd.DataFrame(fragments),
                        }
                        
                        for key, df in dfs.items():
                            df.columns = schema[key]
                            df.index = [index] * len(df)
                            df = df.astype(schema[key])
                            hdf.put(key=key, value=df, format='table', append=True, min_itemsize=100, complevel=1, complib='blosc:lz4')

                        index += 1
                        n += 1
                        if n == limit_per_file:
                            break
                        if progress and index % 10 == 0:
                            pbar.update(10)
                        header.clear()
                        fragments.clear()
                        
    return index

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--msps', type=str, nargs='+')
    parser.add_argument('--hdf', type=str)
    parser.add_argument('--schema', type=str)
    parser.add_argument('--metadata', type=str, default=None)
    parser.add_argument('--progress', type=bool, default=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    
    build_msp_hdf(args.msps, args.hdf, args.schema, args.metadata, args.progress, args.limit)