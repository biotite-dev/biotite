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
                    get_std_guanine, get_std_thymine, get_std_uracil, \
                    norm_vector

_std_adenine, _std_adenine_ring_centers, _std_adenine_hpos = get_std_adenine()
_std_cytosine, _std_cytosine_ring_centers, _std_cytosine_hpos = get_std_cytosine()
_std_guanine, _std_guanine_ring_centers, _std_guanine_hpos = get_std_guanine()  
_std_thymine, _std_thymine_ring_centers, _std_thymine_hpos = get_std_thymine()
_std_uracil, _std_uracil_ring_centers, _std_uracil_hpos = get_std_uracil()

def get_basepairs(array):

    basepair_candidates = _get_proximate_basepair_candidates(array)

    basepairs = []

    for basepair_c in basepair_candidates:
        base1 = array[_filter_residues(array, basepair_c[0], basepair_c[1])]
        base2 = array[_filter_residues(array, basepair_c[2], basepair_c[3])]
        if _check_dssr_criteria([base1, base2]):
            basepairs.append(basepair_c)
    
    return basepairs

def _check_dssr_criteria(basepair):
    #TODO: Increase efficiency by using more np Arrays
    std_bases = [None] * 2
    std_centers = [None] * 2
    std_hpos = [None] * 2
    std_bases_masks = [None] * 2

    vectors = [np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                             [0, 0, 1]], np.float)] * 2

    for i in range(2):
        #TODO Consider Python Pointers
        basepair[i], std_bases[i], std_centers[i], std_hpos[i], std_bases_masks[i] = _match_base(basepair[i])

        if(std_bases[i] == None):
            return False

        vectors[i] = np.vstack((vectors[i], std_centers[i]))

        transformation = superimpose(basepair[i], std_bases[i][std_bases_masks[i]])[1]
        std_bases[i] = superimpose_apply(std_bases[i], transformation)
        trans1, rot, trans2 = transformation

        vectors[i] += trans1
        vectors[i]  = np.dot(rot, vectors[i].T).T
        vectors[i] += trans2
        
        #Normalize z-Vector (orthonormal to xy Plane)
        norm_vector(vectors[i][3,:])

        
    
    #Make sure normal vectors point in same direction

    if np.arccos(np.dot(vectors[0][3,:], vectors[1][3,:])) > np.arccos(np.dot((-1*vectors[0][3,:]), vectors[1][3,:])):
        for i in range(1, 4):
            vectors[0][i,:] = -1*vectors[0][i,:]
    
    #Distance between orgins <= 15 A
    
    if not (distance(vectors[0][0,:], vectors[1][0,:]) <= 15):
        return False

    #Vertical seperation <= 2.5 A

    t = np.linalg.solve(np.vstack( (vectors[0][1,:], vectors[0][2,:], (-1)*vectors[0][3,:]) ).T, (vectors[1][0,:] - vectors[0][0,:]) )[2]
    intersection = vectors[1][0,:] - (t * vectors[0][3,:])

    if not (distance(vectors[1][0,:], intersection) <= 2.5):
        return False
    
    #Angle between normal vectors <= 65°
    
    if not ( ( np.arccos(np.dot(vectors[0][3,:], vectors[1][3,:])) )
                <= ( (65*np.pi)/180 )
            ):
        return False

    #Absence of Stacking
    
    if _check_base_stacking(vectors):
        return False
    
    #Presence of Hydrogen Bonds (Plausability)

    if not _check_hbonds(std_bases, std_hpos):
        return False

    #If no condition was a dealbreaker: Accept Basepair

    return True

def _check_hbonds(std_bases, std_hpos):
    #Accept if Donor-Acceptor Relationship <= 3.5 A exists
    for donor, dmask, acceptor, amask in zip(std_bases, std_hpos, reversed(std_bases), reversed(std_hpos)):
       
        for datom in donor[dmask[0]]:
            for aatom in acceptor[amask[1]]:

                if(distance(aatom.coord, datom.coord) <= 3.5):
                    return True
    
    return False

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
    
    if(wrongdistance == True):
        return False
    
    #Check angle between normal vectors <= 23°

    if not ( ( np.arccos(np.dot(vectors[0][3,:], vectors[1][3,:])) )
                <= ( (23*np.pi)/180 )
            ):
            
            return False
    
    #Determine if angle between normalized_distance vector and one 
    #   normal vector <= 40°
    
    for vector in vectors:
        for norm_dist_vector in norm_dist_vectors:
            
            if ( np.arccos(np.dot(vector[3,:], norm_dist_vector))
                <= ( (40*np.pi)/180 )
            ):
            
                return True
    
    return False

def _match_base(base):
    
    if(base[0].res_name == "A" or base[0].res_name == "DA"):
        std_base = _std_adenine
        std_centers = _std_adenine_ring_centers
        std_hpos = _std_adenine_hpos

    elif(base[0].res_name == "T" or base[0].res_name == "DT"):
        std_base = _std_thymine
        std_centers = _std_thymine_ring_centers
        std_hpos = _std_thymine_hpos

    elif(base[0].res_name == "C" or base[0].res_name == "DC"):
        std_base = _std_cytosine
        std_centers = _std_cytosine_ring_centers
        std_hpos = _std_cytosine_hpos

    elif(base[0].res_name == "G" or base[0].res_name == "DG"):
        std_base = _std_guanine
        std_centers = _std_guanine_ring_centers
        std_hpos = _std_guanine_hpos

    elif(base[0].res_name == "U" or base[0].res_name == "DU"):
        std_base = _std_uracil
        std_centers = _std_uracil_ring_centers
        std_hpos = _std_uracil_hpos 
                
    base = base[np.in1d(base.atom_name, std_base.atom_name)]
    std_base_mask = np.in1d(std_base.atom_name, base.atom_name)

    return base, std_base, std_centers, std_hpos, std_base_mask

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

