# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for basepair identification.
"""

__name__ = "biotite.structure"
__author__ = "Tom Müller"
__all__ = []

import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack, coord
from .filter import filter_nucleotides, _filter_atom_type, _filter_residues
from .celllist import CellList
from .util import distance

def get_basepairs(array):
    basepair_candidates = _get_proximate_basepair_candidates(array)

    basepairs = []

    for basepair_c in basepair_candidates:
        basepair1 = _filter_residues(array, basepair_c[0], basepair_c[1])
        basepair2 = _filter_residues(array, basepair_c[2], basepair_c[3])
        if check_dssr_criteria(basepair1, basepair2):
            basepairs.append(basepair_c)
    
    return basepairs

def check_dssr_criteria(basepair1, basepair2):
    pass



def _get_proximate_basepair_candidates(array, max_cutoff = 15, min_cutoff = 9):
    
    #gets proximate basepairs, where the C1-Sugar-Atoms are within
    # `min_cutoff <= x <= max_cutoff`
    
    array = array[filter_nucleotides(array) 
                    & _filter_atom_type(array, ["C1'", "C1*"])]
    adjacency_matrix = CellList(array, 6.0).create_adjacency_matrix(max_cutoff)
    
    basepair_candidates = []
    
    for ix,iy in np.ndindex(adjacency_matrix.shape):
        if (adjacency_matrix[ix][iy]):
            candidate = [array[ix].res_id, array[ix].chain_id]
            partner = [array[iy].res_id, array[iy].chain_id]
            if ((distance(array[ix].coord, array[iy].coord) > min_cutoff) 
                 & ((partner + candidate) not in basepair_candidates)):
                
                basepair_candidates.append(candidate + partner)
    
    return basepair_candidates