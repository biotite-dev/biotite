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
from .filter import filter_nucleotides, _filter_atom_type
from .celllist import CellList
from .util import distance

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


        
"""
    for atom in array:
        candidates = cell_list.get_atoms(atom.coord, max_cutoff)
        atom_id = [atom.res_id, atom.chain_id]
        
        for candidate in candidates:
            partner_id = [array[candidate].res_id,
                        array[candidate].chain_id]
            if ( (distance(
                        array[int(candidate].coord, atom.coord) > min_cutoff
                        ) 
                 & ((partner_id + atom_id) not in basepair_candidates)
            ):
                
                basepair_candidates.append(atom_id + partner_id)
   
    return basepair_candidates
"""