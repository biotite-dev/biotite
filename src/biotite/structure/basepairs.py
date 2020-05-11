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
                    get_std_guanine, get_std_thymine, get_std_uracil \
                    norm_vector

_std_adenine, _std_adenine_ring_centers = get_std_adenine()
_std_cytosine, _std_cytosine_ring_centers = get_std_cytosine()
_std_guanine, _std_guanine_ring_centers = get_std_guanine()  
_std_thymine, _std_thymine_ring_centers = get_std_thymine()
_std_uracil, _std_uracil_ring_centers = get_std_uracil()

def get_basepairs(array):

    basepair_candidates = _get_proximate_basepair_candidates(array)

    basepairs = []

    for basepair_c in basepair_candidates:
        base1 = _filter_residues(array, basepair_c[0], basepair_c[1])
        base2 = _filter_residues(array, basepair_c[2], basepair_c[3])
        if _check_dssr_criteria([base1, base2]):
            basepairs.append(basepair_c)
    
    return basepairs

def _check_dssr_criteria(basepair):
    #TODO: Increase efficiency by using more np Arrays
    std_bases = [None] * 2
    std_centers = [None] * 2

    vectors = [np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                             [0, 0, 1]], np.float)] * 2

    for i in range(2):
        #TODO Consider Python Pointers
        basepair[i], std_bases[i], std_centers[i] = _match_base(basepair[i])

        if(std_bases[i] == None):
            return False

        vectors[i] = np.vstack((vectors[i], std_centers[i]))

        trans1, rot, trans2 = superimpose(basepair[i], std_bases[i])[1]

        vectors[i] += trans1
        vectors[i]  = np.dot(rot, transformed.coord.T).T
        vectors[i] += trans2

        #Normalize z-Vector (orthonormal to xy Plane)
        norm_vector(vectors[i][3,:])
    
    #Distance between orgins <= 15 A

    if not (distance(vectors[0][0,:], vectors[1][0,:]) <= 15):
        return False

    #Vertical seperation <= 2.5 A

    elif not (abs(vectors[0][0,2] - vectors[1][0,2]) <= 2.5):
        return False
    
    #Angle between normal vectors <= 65°
    
    elif not ( ( np.arccos(np.dot(vectors[0][3,:], vectors[1][3,:])) )
                <= ( (65*np.pi)/180 ):
            )
        return False
    
    #Absence of Stacking
    
    elif _check_base_stacking(vectors):
        return False
    
    #Check if Hydrogen-Bonds are present (possible)
    
    return True

def _check_base_stacking(vectors):
    #checks for the presence of base stacking corresponding to Gabb 1996
    #   DOI: 10.1016/0263-7855(95)00086-0

    #Check for Base-Base Stacking

    #Distance between ring centers <= 4.5 A

    wrongdistance = True
    norm_dist_vectors = []

    for center1 in vectors[0][4:][:]:
        for center2 in vectors[1][4:][:]:
            if (distance(center1, center2) <= 4.5):
                wrongdistance = False
                norm_dist_vectors.append(center2 - center1)
                norm_vector(norm_dist_vectors[-1])            
    
    if(wrongdistance):
        return False
    
    #Check angle between normal vectors <= 23°

    if not ( ( np.arccos(np.dot(vectors[0][3,:], vectors[1][3,:])) )
                <= ( (23*np.pi)/180 )
            ):
            return False
    
    #Determine if angle between normalized_distance vector and one 
    #   normal vector <= 40°
     
    for i in range(2):
        for norm_dist_vector in norm_dist_vectors:
            if not ( ( np.arccos(np.dot((-1*i)*vectors[i][3,:], norm_dist_vector)) )
                <= ( (40*np.pi)/180 )
            ):
            return False

    return True

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

