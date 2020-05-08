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
from .atoms import *
from .superimpose import superimpose, superimpose_apply
from .filter import filter_nucleotides, _filter_atom_type, _filter_residues
from .celllist import CellList
from .util import distance, get_std_adenine, get_std_cytosine, \
                     get_std_guanine, get_std_thymine, get_std_uracil

_std_adenine = get_std_adenine()
_std_cytosine = get_std_cytosine()
_std_guanine = get_std_guanine()  
_std_thymine = get_std_thymine()
_std_uracil = get_std_uracil()

def get_basepairs(array):

    basepair_candidates = _get_proximate_basepair_candidates(array)

    basepairs = []

    for basepair_c in basepair_candidates:
        base1 = _filter_residues(array, basepair_c[0], basepair_c[1])
        base2 = _filter_residues(array, basepair_c[2], basepair_c[3])
        if check_dssr_criteria(base1, base2):
            basepairs.append(basepair_c)
    
    return basepairs

def check_dssr_criteria(base1, base2):
    base1, base1_std = _match_base(base1)
    base2, base2_std = _match_base(base2)

    if((base1_std == None) | (base2_std == None)):
        return False

    _, transformation1 = superimpose(base1, base1_std)
    _, transformation2 = superimpose(base2, base2_std)

    vectors1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float)
    vectors2 = vectors1.copy()

    trans1, rot, trans2 = transformation1

    vectors1 += trans1
    vectors1 = np.dot(rot, transformed.coord.T).T
    vectors1 += trans2
    


def _match_base(base):
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

