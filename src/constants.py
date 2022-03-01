from dataclasses import dataclass

@dataclass(frozen=True)
class MSConstants:
    alphabet = '_ACDEFGHIKLMNOPQRSTVWYcm'
    n_term_ions = 'ab'
    ions = 'aby'
    losses = ('', '-H2O', '-NH3')
    loss_masses =  {
        '-H2O': 18.0105646837,
        '-NH3': 17.02654910101,
    }
    mods = { # this is not general ofc
        ('C','Carbamidomethyl'): 'c',
        ('M','Oxidation'): 'm'
    }
    mod_masses = { # deltas wrt unmodded; not 100% on these...
        'c': 57.02146,
        'm': 15.99491461956
    }
    min_charge = 1 # is this true!?!?
    max_charge = 7
    min_frag_charge = 1
    max_frag_charge = 2
    min_frag_mz = 100
    max_frag_mz = float('inf')
