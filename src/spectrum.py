import numpy as np
from pyteomics.mass import fast_mass

from .constants import MSConstants
C = MSConstants()

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
    fragments = np.zeros((len(sequence)-1,len(C.ions),C.max_frag_charge+1-C.min_frag_charge,len(C.losses)), dtype=np.float32)
    for intensity, bond, ion, charge, loss in zip(intensities, bonds, ions, charges, losses):
        ion = C.ions.index(ion)
        charge = charge - 1
        loss = C.losses.index(loss)
        fragments[bond, ion, charge, loss] = intensity
        
    # treat unsequenced fragments as missing, not zeros
    # and impossible fragments to observed zeros
    fragment_mask = np.zeros_like(fragments,dtype=np.int32)
    fragment_mask[fragments > 0] = 1
    is_invalid_charge = np.arange(C.min_frag_charge,C.max_frag_charge+1) > precursor_charge
    fragment_mask[:,:,is_invalid_charge] = 1
    
   # I am actually not sure about these.
    # they are physically *possible* after all
    # fragment_mzs = fragment_mz_tensor(sequence)
    # fragment_mask[fragment_mzs < C.min_frag_mz] = 1
    # fragment_mask[fragment_mzs > C.max_frag_mz] = 1
    
    return fragments, fragment_mask

def fragment_mz_tensor(sequence):
    masses = np.zeros((len(sequence)-1, len(C.ions), C.max_frag_charge+1-C.min_frag_charge, len(C.losses)), dtype=np.float32)
    for bond in range(len(sequence)-1): # bond+1 !!!
        for i,ion in enumerate(C.ions):
            frag = sequence[:bond+1] if ion in C.n_term_ions else sequence[bond+1:]
            for j,charge in enumerate(range(C.min_frag_charge,C.max_frag_charge+1)):
                for k,loss in enumerate(C.losses):
                    masses[bond,i,j,k] = compute_peptide_mz(frag, ion, charge, loss)
    return masses
    
def compute_peptide_mz(sequence, ion, charge, loss):
    if charge == 0 or sequence is None:
        return np.nan
    mz = fast_mass(sequence.upper(), ion, charge)
    for aa in sequence:
        if aa in C.mod_masses:
            mz += C.mod_masses[aa] / charge
    if loss != '':
        mz -= C.loss_masses[loss] / charge
    return mz

def transform_spectrum(item):
    index = item['index']
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
    
    fragments, fragment_mask = tensorize_fragments(
        intensities, 
        bonds, 
        ions,
        charges,
        losses,
        modded_sequence,
        precursor_charge
    )
    
    item = {}
    item['index'] = index
    item['x'] = codes
    item['y'] = fragments
    item['x_mask'] = np.ones_like(codes, dtype=np.int32)
    item['y_mask'] = fragment_mask
    item['charge'] = precursor_charge
    item['collision_energy'] = collision_energy
    item['sequence'] = modded_sequence
    
    return item
