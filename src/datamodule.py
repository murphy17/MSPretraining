import numpy as np
from pyteomics.mass import fast_mass

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .torch_helpers import RejectionSampler, zero_padding_collate, cache_path, PandasHDFDataset

################################################################################
# class to hold all the hard coded parameters in one place
################################################################################

class MSConstants:
    def __init__(self):
        self.alphabet = '_ACDEFGHIKLMNOPQRSTVWYcm'
#         self.n_term_ions = 'abc'
#         self.ions = 'abcxyz'
        self.n_term_ions = 'ab'
        self.ions = 'aby'
        self.losses = ('', '-H2O', '-NH3')
        self.loss_masses =  {
            '-H2O': 18.0105646837,
            '-NH3': 17.02654910101,
        }
        self.mods = { # this is not general ofc
            ('C','Carbamidomethyl'): 'c',
            ('M','Oxidation'): 'm'
        }
        self.mod_masses = { # deltas wrt unmodded; not 100% on these...
            'c': 57.02146,
            'm': 15.99491461956
        }
        self.max_charge = 7
        self.max_frag_charge = 2
        self.min_frag_mz = 100
        self.max_frag_mz = float('inf')

C = MSConstants()

################################################################################
# bunch of helper methods for processing spectra
################################################################################

def index_fragment_bonds(ions, lengths, precursor_length):
    bonds = [length-1 if ion in C.n_term_ions else precursor_length-length-1 for ion, length in zip(ions, lengths)]
    return bonds

def modify_sequence(sequence, mod_pos, mod_name):
    sequence = list(sequence)
    for pos, mod in zip(mod_pos, mod_name):
        sequence[pos] = C.mods[(sequence[pos], mod)]
    return ''.join(sequence)

def encode_sequence(sequence):
    codes = np.array([C.alphabet.index(aa) for aa in sequence])
    return codes

def tensorize_fragments(intensities, bonds, ions, charges, losses, sequence, precursor_charge):
    fragments = np.zeros((len(sequence)-1,len(C.ions),C.max_frag_charge,len(C.losses)), dtype=np.float32)
    for intensity, bond, ion, charge, loss in zip(intensities, bonds, ions, charges, losses):
        ion = C.ions.index(ion)
        charge = charge - 1
        loss = C.losses.index(loss)
        fragments[bond, ion, charge, loss] = intensity
        
    # treat unsequenced fragments as missing, not zeros
    # and impossible fragments to observed zeros
    fragment_mask = np.zeros_like(fragments,dtype=np.int32)
    fragment_mzs = fragment_mz_tensor(sequence)
    fragment_mask[fragment_mzs < C.min_frag_mz] = 1
    fragment_mask[fragment_mzs > C.max_frag_mz] = 1
    fragment_mask[:,:,np.arange(C.max_frag_charge) > precursor_charge-1] = 1
    
    return fragments, fragment_mask

def fragment_mz_tensor(sequence):
    masses = np.zeros((len(sequence)-1, len(C.ions), C.max_frag_charge, len(C.losses)), dtype=np.float32)
    for bond in range(len(sequence)-1): # bond+1 !!!
        for i,ion in enumerate(C.ions):
            frag = sequence[:bond+1] if ion in C.n_term_ions else sequence[bond+1:]
            for j,charge in enumerate(range(1,C.max_frag_charge+1)):
                for k,loss in enumerate(C.losses):
                    masses[bond,i,j,k] = compute_peptide_mz(frag, ion, charge, loss)
    return masses
    
def compute_peptide_mz(sequence, ion, charge, loss):
    if charge == 0 or sequence is None:
        return np.nan
    mz = fast_mass(sequence.upper(), ion, charge)
    for aa in sequence:
        if aa in C.mod_masses:
            mz += C.mod_masses[aa]
    if loss != '':
        mz -= C.loss_masses[loss]
    return mz

def transform_spectrum(item):
    sequence = item['Spectrum']['sequence'][0]
    precursor_charge = item['Spectrum']['charge'][0]
    precursor_length = len(sequence)
    collision_energy = item['Spectrum']['collision_energy'][0]

    intensities = item['Fragment']['intensity']
    ions = item['Fragment']['ion']
    lengths = item['Fragment']['length']
    charges = item['Fragment']['charge']
    losses = item['Fragment']['loss']

    mod_pos = item['Modification']['position']
    mod_aa = item['Modification']['residue']
    mod = item['Modification']['modification']
    
    modded_sequence = modify_sequence(sequence, mod_pos, mod)

    bonds = index_fragment_bonds(ions, lengths, precursor_length)
    codes = encode_sequence(modded_sequence)
    
    fragments, fragment_mask = tensorize_fragments(intensities, bonds, ions, charges, losses, modded_sequence, precursor_charge)
    
    item = {}
    item['x'] = codes
    item['y'] = fragments
    item['x_mask'] = np.ones_like(codes, dtype=np.int32)
    item['y_mask'] = fragment_mask
    item['charge'] = precursor_charge
    item['collision_energy'] = collision_energy
    item['sequence'] = modded_sequence
    
    return item


################################################################################
# a lightning data module
################################################################################

class MSDataModule(LightningDataModule):
    def __init__(self, hdf_path, batch_size, train_filter, valid_filter, num_workers=1, cache_dir=None):
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