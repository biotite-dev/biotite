# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["standardize_order"]

import numpy as np
from .atoms import residue
from ..residues import get_residue_starts


_atom_name_cache = {}


def standardize_order(atoms):
    """
    Get an index array for an input :class:`AtomArray` or
    :class:`AtomArrayStack` that reorders the atoms for each residue
    to obtain the standard *RCSB PDB* atom order.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or AtomArrayStack, shape=(m,n)
        Input structure with atoms that are potentially not in the
        *standard* order.
    
    Returns
    indices : ndarray, dtype=int, shape=(n,)
        When this index array is applied on the input `atoms`,
        the atoms for each residue are reordered to obtain the
        standard *RCSB PDB* atom order.
    """
    reordered_indices = np.zeros(atoms.array_length(), dtype=int)

    starts = get_residue_starts(atoms, add_exclusive_stop=True)
    for i in range(len(starts)-1):
        start = starts[i]
        stop = starts[i+1]

        res_name = atoms.res_name[start]
        standard_atom_names = _atom_name_cache.get(res_name)
        if standard_atom_names is None:
            standard_atom_names = residue(res_name).atom_name
            _atom_name_cache[res_name] = standard_atom_names
        
        reordered_indices[start : stop] = _reorder(
            atoms.atom_name[start : stop], standard_atom_names
        ) + start

    return reordered_indices


def _reorder(origin, target):
    indices = np.zeros(len(origin), dtype=int)
    i = 0
    for e in target:
        hits = np.where(e == origin)[0]
        if len(hits) == 1:
           indices[i] = hits[0]
           i += 1
        elif len(hits) == 0:
            # Target atom is not in array of original atoms
            pass
        else:
            # Original atoms contain a duplicate
            raise ValueError("Original structure contains duplicate atoms")
    if i < len(origin):
        raise ValueError("Target structure misses atoms from original structure")
    return indices