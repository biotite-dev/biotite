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
from .hbond import hbond
from .util import distance, get_std_adenine, get_std_cytosine, \
                    get_std_guanine, get_std_thymine, get_std_uracil, \
                    norm_vector

_std_adenine, _std_adenine_ring_centers, \
        _std_adenine_hpos = get_std_adenine()
_std_cytosine, _std_cytosine_ring_centers, \
        _std_cytosine_hpos = get_std_cytosine()
_std_guanine, _std_guanine_ring_centers, \
        _std_guanine_hpos = get_std_guanine()  
_std_thymine, _std_thymine_ring_centers, \
        _std_thymine_hpos = get_std_thymine()
_std_uracil, _std_uracil_ring_centers, \
        _std_uracil_hpos = get_std_uracil()

_adenine_like = ["A", "DA"]
_thymine_like = ["T", "DT"]
_cytosine_like = ["C", "DC"]
_guanine_like = ["G", "DG"]
_uracil_like = ["U", "DU"]

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

    p_bases = [None] * 2
    std_hpos = [None] * 2
    vectors = [None] * 2
    hydrogens = np.ones(2, dtype=bool)

    for i in range(2):
        
        base_tuple = _match_base(basepair[i])

        if(base_tuple == None):
            return False
        
        else:
            p_bases[i], std_hpos[i], hydrogens[i], vectors[i] = base_tuple

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
    
    #Presence of Hydrogen Bonds (Plausability if no Hydrogens)
    ba = (p_bases[0] + p_bases[1])
    if (np.all(hydrogens)):
        
        if(len(hbond(ba, np.ones_like(ba, dtype=bool), 
                np.ones_like(ba, dtype=bool))) == 0):
                return False
               
    elif not _check_hbonds(p_bases, std_hpos):
        
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
    #Matches a nucleotide to a standard base
    #Returns: 
    #ret_base : The base or if the base atoms are incomplete a
    #               superimposed standard base
    #ret_hpos : A list of size 2 containing boolean masks. 
    #               Pos 0 contains the het_atoms that act as H-Donors
    #               Pos 1 contains the het_atoms that act as H-Acceptors
    #contains_hydrogens : A boolean; if True the base contains H-Atoms
    #vectors : A set of std_vectors (Origin, Orthonormal-Base-Vectors, 
    #               Ring-Centers) transformed onto the
    #               nucleotides coordinates   

    ret_hpos = [None] * 2

    vector = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                             [0, 0, 1]], np.float)

    #Check Base Type

    if(base[0].res_name in _adenine_like):
        std_base = _std_adenine
        std_centers = _std_adenine_ring_centers
        std_hpos = _std_adenine_hpos

    elif(base[0].res_name in _thymine_like):
        std_base = _std_thymine
        std_centers = _std_thymine_ring_centers
        std_hpos = _std_thymine_hpos

    elif(base[0].res_name in _cytosine_like):
        std_base = _std_cytosine
        std_centers = _std_cytosine_ring_centers
        std_hpos = _std_cytosine_hpos

    elif(base[0].res_name in _guanine_like):
        std_base = _std_guanine
        std_centers = _std_guanine_ring_centers
        std_hpos = _std_guanine_hpos

    elif(base[0].res_name in _uracil_like):
        std_base = _std_uracil
        std_centers = _std_uracil_ring_centers
        std_hpos = _std_uracil_hpos 
    
    else:
        #TODO: Throw Warning
        return None

    #Check if the structure uses PDBv3 or PDBv2 atom nomenclature
    
    if( np.sum(np.in1d(std_base[1].atom_name, base.atom_name))
            > np.sum(np.in1d(std_base[0].atom_name, base.atom_name))
    ):
        std_base = std_base[1]
    else:
        std_base = std_base[0]

    #Add the Ring Centers onto the array of vectors to be transformed

    vector = np.vstack((vector, std_centers))
    
    #Match the selected std_base to the base

    fitted, transformation = superimpose(
                        base[np.in1d(base.atom_name, std_base.atom_name)],
                        std_base[np.in1d(std_base.atom_name, base.atom_name)]
                                        )

    #Investigate the completeness of the base:
    #       A length difference of zero means the base contains all
    #       atoms of the std_base
          
    length_difference = len(std_base) - len(fitted) 
    
    #Transform the vectors

    trans1, rot, trans2 = transformation

    vector += trans1
    vector  = np.dot(rot, vector.T).T
    vector += trans2
    
    #Normalise the transformed orthogonal base vectors

    for i in range(1, 4):
        norm_vector(vector[i,:])
    
    #If the base is incomplete but contains 3 or more atoms of the 
    #   std_base, transform the complete std_base and use it to
    #   approximate the base.

    if(length_difference > 0 and len(fitted) >= 3):
        #TODO: Throw Warning
        ret_base = superimpose_apply(std_base, transformation)
        ret_hpos = std_hpos
        contains_hydrogens = False
    
    #If the base is incomplete and conatains less than 3 atoms of the 
    #   std_base throw warning

    elif (length_difference > 0):
        #TODO: Throw Warning
        return None

    #if the base is complete use the base for further calculations    
    else:

        mask = np.ones(len(base), dtype=bool)
        
        # Generate a boolean mask containing only the base atoms,
        #   disregarding the sugar atoms and the phosphate backbone

        for i in range(len(base)):
            if( ("'" in base[i].atom_name) or ("*" in base[i].atom_name) or
                (   
                    (base[i].atom_name not in std_base.atom_name) and
                    (base[i].element != "H") 
                )
            ):
                mask[i] = False
        
        #Generate a boolaean mask for the hydrogen donors and acceptors

        for i in range(2):
            ret_hpos[i] = _filter_atom_type(base[mask], 
                                std_base[std_hpos[i]].atom_name)

        #Check if the base contains Hydrogens
        if ("H" in base.element[mask]):
            contains_hydrogens = True
            ret_base = base[mask]
                    
        else:
            ret_base = base[mask]
        
    return ret_base, ret_hpos, contains_hydrogens, vector

def _get_proximate_basepair_candidates(array, max_cutoff = 15, min_cutoff = 9):
    
    #gets proximate basepairs, where the C1-Sugar-Atoms are within
    # `min_cutoff <= x <= max_cutoff`
    
    c1sugars = array[filter_nucleotides(array) 
                    & _filter_atom_type(array, ["C1'", "C1*"])]
    adj_matrix = CellList(c1sugars, 6.0).create_adjacency_matrix(max_cutoff)
    
    basepair_candidates = []
    
    for ix,iy in np.ndindex(adj_matrix.shape):
        if (adj_matrix[ix][iy]):
            candidate = [c1sugars[ix].res_id, c1sugars[ix].chain_id]
            partner = [c1sugars[iy].res_id, c1sugars[iy].chain_id]
            if ((distance(c1sugars[ix].coord, c1sugars[iy].coord) > min_cutoff) 
                 & ((partner + candidate) not in basepair_candidates)):
                
                basepair_candidates.append(candidate + partner)
    
    return basepair_candidates

