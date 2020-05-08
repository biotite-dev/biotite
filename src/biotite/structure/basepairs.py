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
    _std_adenine = _get_std_adenine()
    _std_cytosine = _get_std_cytosine()
    _std_guanine = _get_std_guanine()  
    _std_thymine = _get_std_thymine()
    _std_uracil = _get_std_uracil()
    
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

def _get_std_adenine():
    atom1 = Atom([-2.479, 5.346, 0.000], atom_id="1", atom_name="C1′", res_id="A")
    atom2 = Atom([-1.291, 4.498, 0.000], atom_id="2", atom_name="N9", res_id="A")
    atom3 = Atom([0.024, 4.897, 0.000], atom_id="3", atom_name="C8", res_id="A")
    atom4 = Atom([0.877, 3.902, 0.000], atom_id="4", atom_name="N7", res_id="A")
    atom5 = Atom([0.071, 2.771, 0.000], atom_id="5", atom_name="C5", res_id="A")
    atom6 = Atom([0.369, 1.398, 0.000], atom_id="6", atom_name="C6", res_id="A")
    atom7 = Atom([1.611, 0.909, 0.000], atom_id="7", atom_name="N6", res_id="A")
    atom8 = Atom([-0.668, 0.532, 0.000], atom_id="8", atom_name="N1", res_id="A")
    atom9 = Atom([-1.912, 1.023, 0.000], atom_id="9", atom_name="C2", res_id="A")
    atom10 = Atom([-2.320, 2.290, 0.000], atom_id="10", atom_name="N3", res_id="A")
    atom11 = Atom([-1.267, 3.124, 0.000], atom_id="11", atom_name="C4", res_id="A")
    adenine = array[atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9, atom10, atom11] 
    return adenine

def _get_std_cytosine():
    atom1 = Atom([-2.477, 5.402, 0.000], atom_id="1", atom_name="C1′", res_id="C")
    atom2 = Atom([-1.285, 4.542, 0.000], atom_id="2", atom_name="N1", res_id="C")
    atom3 = Atom([-1.472, 3.158, 0.000], atom_id="3", atom_name="C2", res_id="C")
    atom4 = Atom([-2.628, 2.709, 0.000], atom_id="4", atom_name="O2", res_id="C")
    atom5 = Atom([-0.391, 2.344, 0.000], atom_id="5", atom_name="N3", res_id="C")
    atom6 = Atom([0.837, 2.868, 0.000], atom_id="6", atom_name="C4", res_id="C")
    atom7 = Atom([1.875, 2.027, 0.000], atom_id="7", atom_name="N4", res_id="C")
    atom8 = Atom([1.056, 4.275, 0.000], atom_id="8", atom_name="C5", res_id="C")
    atom9 = Atom([-0.023, 5.068, 0.000], atom_id="9", atom_name="C6", res_id="C")
    cytosine = array[atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9] 
    return cytosine

def _get_std_guanine():
    atom1 = Atom([-2.477, 5.399, 0.000], atom_id="1", atom_name="C1′", res_id="G")
    atom2 = Atom([-1.289, 4.551, 0.000], atom_id="2", atom_name="N9", res_id="G")
    atom3 = Atom([0.023, 4.962, 0.000], atom_id="3", atom_name="C8", res_id="G")
    atom4 = Atom([0.870, 3.969, 0.000], atom_id="4", atom_name="N7", res_id="G")
    atom5 = Atom([0.071, 2.833, 0.000], atom_id="5", atom_name="C5", res_id="G")
    atom6 = Atom([0.424, 1.460, 0.000], atom_id="6", atom_name="C6", res_id="G")
    atom7 = Atom([1.554, 0.955, 0.000], atom_id="7", atom_name="O6", res_id="G")
    atom8 = Atom([-0.700, 0.641, 0.000], atom_id="8", atom_name="N1", res_id="G")
    atom9 = Atom([-1.999, 1.087, 0.000], atom_id="9", atom_name="C2", res_id="G")
    atom10 = Atom([-2.949, 0.139, -0.001], atom_id="10", atom_name="N2", res_id="G")
    atom11 = Atom([-2.342, 2.364, 0.001], atom_id="11", atom_name="N3", res_id="G")
    atom12 = Atom([-1.265, 3.177, 0.000], atom_id="12", atom_name="C4", res_id="G")
    guanine = array[atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9, atom10, atom11, atom12] 
    return guanine

def _get_std_thymine():
    atom1 = Atom([-2.481, 5.354, 0.000], atom_id="1", atom_name="C1′", res_id="T")
    atom2 = Atom([-1.284, 4.500, 0.000], atom_id="2", atom_name="N1", res_id="T")
    atom3 = Atom([-1.462, 3.135, 0.000], atom_id="3", atom_name="C2", res_id="T")
    atom4 = Atom([-2.562, 2.608, 0.000], atom_id="4", atom_name="O2", res_id="T")
    atom5 = Atom([-0.298, 2.407, 0.000], atom_id="5", atom_name="N3", res_id="T")
    atom6 = Atom([0.994, 2.897, 0.000], atom_id="6", atom_name="C4", res_id="T")
    atom7 = Atom([1.944, 2.119, 0.000], atom_id="7", atom_name="O4", res_id="T")
    atom8 = Atom([1.106, 4.338, 0.000], atom_id="8", atom_name="C5", res_id="T")
    atom9 = Atom([2.466, 4.961, 0.001], atom_id="9", atom_name="C5M", res_id="T")
    atom10 = Atom([-0.024, 5.057, 0.000], atom_id="10", atom_name="C6", res_id="T")
    thymine = array[atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9, atom10] 
    return thymine

def _get_std_uracil():
    atom1 = Atom([-2.481, 5.354, 0.000], atom_id="1", atom_name="C1′", res_id="U")
    atom2 = Atom([-1.284, 4.500, 0.000], atom_id="2", atom_name="N1", res_id="U")
    atom3 = Atom([-1.462, 3.131, 0.000], atom_id="3", atom_name="C2", res_id="U")
    atom4 = Atom([-2.563, 2.608, 0.000], atom_id="4", atom_name="O2", res_id="U")
    atom5 = Atom([-0.302, 2.397, 0.000], atom_id="5", atom_name="N3", res_id="U")
    atom6 = Atom([0.989, 2.884, 0.000], atom_id="6", atom_name="C4", res_id="U")
    atom7 = Atom([1.935, 2.094, -0.001], atom_id="7", atom_name="O4", res_id="U")
    atom8 = Atom([1.089, 4.311, 0.000], atom_id="8", atom_name="C5", res_id="U")
    atom9 = Atom([-0.024, 5.053, 0.000], atom_id="9", atom_name="C6", res_id="U")
    uracil = array[atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9] 
    return uracil


