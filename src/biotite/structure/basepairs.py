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
from .filter import filter_nucleotides
from .celllist import CellList
from .hbond import hbond
from .error import IncompleteStructureWarning, UnexpectedStructureWarning
from .util import distance, norm_vector

"""
The following functions describe the bases adenine, cytosine, thymine,
guanine and uracil in standard coordinates as described by (Wilma, 2001)
TODO: DOI

They Return:

The bases as list:
    0: AtomArray with nomenclature of PDB File Format V2
    1: AtomArray with nomenclature of PDB File Format V3

The center-coordinates of the aromatic rings as list:
    0: Pyrimidine Ring
    1: Imidazole Ring (if present)

The hydrogen bond donors and acceptors as list
    0: Heteroatoms that are bound to a hydrogen that can act as a donor
    1: Heteroatoms that can act as an acceptor
"""


def _get_std_adenine():
    atom1 = Atom([-2.479, 5.346, 0.000], atom_name="C1*", res_name="A")
    atom2 = Atom([-1.291, 4.498, 0.000], atom_name="N9", res_name="A")
    atom3 = Atom([0.024, 4.897, 0.000], atom_name="C8", res_name="A")
    atom4 = Atom([0.877, 3.902, 0.000], atom_name="N7", res_name="A")
    atom5 = Atom([0.071, 2.771, 0.000], atom_name="C5", res_name="A")
    atom6 = Atom([0.369, 1.398, 0.000], atom_name="C6", res_name="A")
    atom7 = Atom([1.611, 0.909, 0.000], atom_name="N6", res_name="A")
    atom8 = Atom([-0.668, 0.532, 0.000], atom_name="N1", res_name="A")
    atom9 = Atom([-1.912, 1.023, 0.000], atom_name="C2", res_name="A")
    atom10 = Atom([-2.320, 2.290, 0.000], atom_name="N3", res_name="A")
    atom11 = Atom([-1.267, 3.124, 0.000], atom_name="C4", res_name="A")
    adenine = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, 
         atom9, atom10, atom11]
                )
    adenine_pdbv3 = adenine.copy()
    adenine_pdbv3.atom_name[[0]] = ["C1'"]

    # Calculate the coordinates of the aromatic ring centers.
    pyrimidine_center = np.mean(
        [atom5.coord, atom6.coord, atom8.coord, 
         atom9.coord, atom10.coord, atom11.coord], axis=-2
                            )
    imidazole_center = np.mean(
        [atom2.coord, atom3.coord, atom4.coord,
         atom5.coord, atom11.coord], axis=-2
                            )

    # Create boolean masks for the AtomArray containing the bases 
    # heteroatoms which (or the usually attached hydrogens) can act as
    # Hydrogen Bond Donors or Acceptors
    hbond_donors = np.zeros(11, dtype=bool)
    hbond_d = [1, 6]
    hbond_donors[hbond_d] = np.ones(len(hbond_d), dtype=bool)
    
    hbond_acceptors = np.zeros(11, dtype=bool)
    hbond_a = [1, 3, 6, 7, 9]
    hbond_acceptors[hbond_a] = np.ones(len(hbond_a), dtype=bool)

    return [adenine, adenine_pdbv3], [pyrimidine_center, imidazole_center], \
           [hbond_donors, hbond_acceptors]

def _get_std_cytosine():
    atom1 = Atom([-2.477, 5.402, 0.000], atom_name="C1*", res_name="C")
    atom2 = Atom([-1.285, 4.542, 0.000], atom_name="N1", res_name="C")
    atom3 = Atom([-1.472, 3.158, 0.000], atom_name="C2", res_name="C")
    atom4 = Atom([-2.628, 2.709, 0.000], atom_name="O2", res_name="C")
    atom5 = Atom([-0.391, 2.344, 0.000], atom_name="N3", res_name="C")
    atom6 = Atom([0.837, 2.868, 0.000], atom_name="C4", res_name="C")
    atom7 = Atom([1.875, 2.027, 0.000], atom_name="N4", res_name="C")
    atom8 = Atom([1.056, 4.275, 0.000], atom_name="C5", res_name="C")
    atom9 = Atom([-0.023, 5.068, 0.000], atom_name="C6", res_name="C")
    cytosine = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9]
                    )
    cytosine_pdbv3 = cytosine.copy()
    cytosine_pdbv3.atom_name[[0]] = ["C1'"]

    # Calculate the coordinates of the aromatic ring center.
    pyrimidine_center = np.mean(
        [atom2.coord, atom3.coord, atom5.coord,
         atom6.coord, atom8.coord, atom9.coord], axis=-2
                            )
    
    # Create boolean masks for the AtomArray containing the bases` 
    # heteroatoms which (or the usually attached hydrogens) can act as
    # Hydrogen Bond Donors or Acceptors respectively.
    hbond_donors = np.zeros(9, dtype=bool)
    hbond_d = [1, 6]
    hbond_donors[hbond_d] = np.ones(len(hbond_d), dtype=bool)
    
    hbond_acceptors = np.zeros(9, dtype=bool)
    hbond_a = [1, 3, 4, 6]
    hbond_acceptors[hbond_a] = np.ones(len(hbond_a), dtype=bool)

    return [cytosine, cytosine_pdbv3], [pyrimidine_center], \
           [hbond_donors, hbond_acceptors]

def _get_std_guanine():
    atom1 = Atom([-2.477, 5.399, 0.000], atom_name="C1*", res_name="G")
    atom2 = Atom([-1.289, 4.551, 0.000], atom_name="N9", res_name="G")
    atom3 = Atom([0.023, 4.962, 0.000], atom_name="C8", res_name="G")
    atom4 = Atom([0.870, 3.969, 0.000], atom_name="N7", res_name="G")
    atom5 = Atom([0.071, 2.833, 0.000], atom_name="C5", res_name="G")
    atom6 = Atom([0.424, 1.460, 0.000], atom_name="C6", res_name="G")
    atom7 = Atom([1.554, 0.955, 0.000], atom_name="O6", res_name="G")
    atom8 = Atom([-0.700, 0.641, 0.000], atom_name="N1", res_name="G")
    atom9 = Atom([-1.999, 1.087, 0.000], atom_name="C2", res_name="G")
    atom10 = Atom([-2.949, 0.139, -0.001], atom_name="N2", res_name="G")
    atom11 = Atom([-2.342, 2.364, 0.001], atom_name="N3", res_name="G")
    atom12 = Atom([-1.265, 3.177, 0.000], atom_name="C4", res_name="G")
    guanine = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, 
         atom9, atom10, atom11, atom12]
                )
    guanine_pdbv3 = guanine.copy()
    guanine_pdbv3.atom_name[[0]] = ["C1'"]

    # Calculate the coordinates of the aromatic ring centers.
    pyrimidine_center = np.mean(
        [atom5.coord, atom6.coord, atom8.coord,
         atom9.coord, atom11.coord, atom12.coord], axis=-2
                            )
    imidazole_center = np.mean(
        [atom2.coord, atom3.coord, atom4.coord,
         atom5.coord, atom12.coord], axis=-2
                            )

    # Create boolean masks for the AtomArray containing the bases` 
    # heteroatoms which (or the usually attached hydrogens) can act as
    # Hydrogen Bond Donors or Acceptors respectively.
    hbond_donors = np.zeros(12, dtype=bool)
    hbond_d = [1, 7, 9]
    hbond_donors[hbond_d] = np.ones(len(hbond_d), dtype=bool)
    
    hbond_acceptors = np.zeros(12, dtype=bool)
    hbond_a = [1, 3, 6, 7, 9, 10]
    hbond_acceptors[hbond_a] = np.ones(len(hbond_a), dtype=bool)
    
    return [guanine, guanine_pdbv3], [pyrimidine_center, imidazole_center], \
           [hbond_donors, hbond_acceptors]

def _get_std_thymine():
    atom1 = Atom([-2.481, 5.354, 0.000], atom_name="C1*", res_name="T")
    atom2 = Atom([-1.284, 4.500, 0.000], atom_name="N1", res_name="T")
    atom3 = Atom([-1.462, 3.135, 0.000], atom_name="C2", res_name="T")
    atom4 = Atom([-2.562, 2.608, 0.000], atom_name="O2", res_name="T")
    atom5 = Atom([-0.298, 2.407, 0.000], atom_name="N3", res_name="T")
    atom6 = Atom([0.994, 2.897, 0.000], atom_name="C4", res_name="T")
    atom7 = Atom([1.944, 2.119, 0.000], atom_name="O4", res_name="T")
    atom8 = Atom([1.106, 4.338, 0.000], atom_name="C5", res_name="T")
    atom9 = Atom([2.466, 4.961, 0.001], atom_name="C5M", res_name="T")
    atom10 = Atom([-0.024, 5.057, 0.000], atom_name="C6", res_name="T")

    pyrimidine_center = np.mean([atom2.coord, atom3.coord, atom5.coord,
                                    atom6.coord, atom8.coord, atom10.coord],
                                    axis=-2
                            )

    hbond_donors = np.zeros(10, dtype=bool)
    hbond_d = [1, 4]
    hbond_donors[hbond_d] = np.ones(len(hbond_d), dtype=bool)
    
    hbond_acceptors = np.zeros(10, dtype=bool)
    hbond_a = [1, 3, 4, 6]
    hbond_acceptors[hbond_a] = np.ones(len(hbond_a), dtype=bool)

    thymine = array([atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, 
                        atom9, atom10]
                )

    v3 = thymine.copy()
    v3.atom_name[[0, 8]] = ["C1'", "C7"]

    return [thymine, v3], [pyrimidine_center], [hbond_donors, hbond_acceptors]

def _get_std_uracil():
    atom1 = Atom([-2.481, 5.354, 0.000], atom_name="C1*", res_name="U")
    atom2 = Atom([-1.284, 4.500, 0.000], atom_name="N1", res_name="U")
    atom3 = Atom([-1.462, 3.131, 0.000], atom_name="C2", res_name="U")
    atom4 = Atom([-2.563, 2.608, 0.000], atom_name="O2", res_name="U")
    atom5 = Atom([-0.302, 2.397, 0.000], atom_name="N3", res_name="U")
    atom6 = Atom([0.989, 2.884, 0.000], atom_name="C4", res_name="U")
    atom7 = Atom([1.935, 2.094, -0.001], atom_name="O4", res_name="U")
    atom8 = Atom([1.089, 4.311, 0.000], atom_name="C5", res_name="U")
    atom9 = Atom([-0.024, 5.053, 0.000], atom_name="C6", res_name="U")
    
    pyrimidine_center = np.mean([atom2.coord, atom3.coord, atom5.coord,
                                    atom6.coord, atom8.coord, atom9.coord],
                                    axis=-2
                            )

    hbond_donors = np.zeros(9, dtype=bool)
    hbond_d = [1, 4]
    hbond_donors[hbond_d] = np.ones(len(hbond_d), dtype=bool)
    
    hbond_acceptors = np.zeros(9, dtype=bool)
    hbond_a = [1, 3, 4, 6]
    hbond_acceptors[hbond_a] = np.ones(len(hbond_a), dtype=bool)

    uracil = array([atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, 
                        atom9]
                )

    v3 = uracil.copy()
    v3.atom_name[[0]] = ["C1'"]

    return [uracil, v3], [pyrimidine_center], [hbond_donors, hbond_acceptors]





_std_adenine, _std_adenine_ring_centers, \
        _std_adenine_hpos = _get_std_adenine()
_std_cytosine, _std_cytosine_ring_centers, \
        _std_cytosine_hpos = _get_std_cytosine()
_std_guanine, _std_guanine_ring_centers, \
        _std_guanine_hpos = _get_std_guanine()  
_std_thymine, _std_thymine_ring_centers, \
        _std_thymine_hpos = _get_std_thymine()
_std_uracil, _std_uracil_ring_centers, \
        _std_uracil_hpos = _get_std_uracil()

_adenine_like = ["A", "DA"]
_thymine_like = ["T", "DT"]
_cytosine_like = ["C", "DC"]
_guanine_like = ["G", "DG"]
_uracil_like = ["U", "DU"]

#TODO: Add Doc

def get_basepairs(array, min_atoms = 3):

    basepair_candidates = _get_proximate_basepair_candidates(array)
    
    basepairs = []

    for basepair_c in basepair_candidates:
        base1 = array[_filter_residues(array, basepair_c[0], basepair_c[1])]
        base2 = array[_filter_residues(array, basepair_c[2], basepair_c[3])]
        if _check_dssr_criteria([base1, base2], min_atoms):
            basepairs.append(basepair_c)
    
    return basepairs

def _check_dssr_criteria(basepair, min_atoms):
    #Returns True if the basepair meets the dssr critera, False if not

    p_bases = [None] * 2
    std_hpos = [None] * 2
    vectors = [None] * 2
    hydrogens = np.zeros(2, dtype=bool)

    #Generate data for each base neccesary for analysis

    for i in range(2):
        
        base_tuple = _match_base(basepair[i], min_atoms)

        if(base_tuple is None):
            return False
        
        p_bases[i], std_hpos[i], hydrogens[i], vectors[i] = base_tuple

    #(i) Distance between orgins <= 15 A
   
    if not (distance(vectors[0][0,:], vectors[1][0,:]) <= 15):
        return False
    
    #(ii) Vertical seperation <= 2.5 A
        
        #Find the intercept between the plane of base zero and a
        #line consisting of the origin of base one and normal vector
        #of base zero
    
    t = np.linalg.solve(np.vstack( (vectors[0][1,:], vectors[0][2,:],
                                   (-1)*vectors[0][3,:]) 
                                ).T,
                         (vectors[1][0,:] - vectors[0][0,:])
                    )[0]
    intercept = vectors[1][0,:] + (t * vectors[0][3,:])

        #Vertical seperation is the distance of the origin of base one
        #and the intercept descibed above
    #print(vectors[0][1:4,:])
    #print(vectors[1][1:4,:])
    #print(distance(vectors[1][0,:], intercept))
    print(str(basepair[0][0].res_id) + str(basepair[0][0].chain_id) + " und " + str(basepair[1][0].res_id) + str(basepair[1][0].chain_id)) 
    if not (distance(vectors[1][0,:], intercept) <= 2.5):
        return False
      
    #(iii) Angle between normal vectors <= 65°
    
    print(np.arccos(np.dot(vectors[0][3,:], vectors[1][3,:]))*(180/np.pi))
    if not ( ( np.arccos(np.dot(vectors[0][3,:], vectors[1][3,:])) )
                >= ( (115*np.pi)/180)
            ):
        return False
    
    #(iv) Absence of Stacking
    
    if _check_base_stacking(vectors):
        return False
    
    #(v) Presence of Hydrogen Bonds
    
    if (np.all(hydrogens)):
        #if the structure contains hydrogens, check for bonds
        if(len(hbond(p_bases[0] + p_bases[1],
                     np.ones_like(p_bases[0] + p_bases[1], dtype=bool), 
                     np.ones_like(p_bases[0] + p_bases[1], dtype=bool)
                    )
                ) == 0):
                return False
               
    elif not _check_hbonds(p_bases, std_hpos):
        #if the structure does not contain hydrogens, check for
        #plausability of Hydrogen Bonds
        return False

    #If no condition was a dealbreaker: Accept Basepair

    return True

def _check_hbonds(std_bases, std_hpos):
    #Accept if het_Donor-het_Acceptor Relationship <= 3.5 A exists
    #Definition from https://proteopedia.org/wiki/index.php/Hydrogen_bonds

    for donor, dmask, acceptor, amask in zip(
                std_bases, std_hpos, reversed(std_bases), reversed(std_hpos)
                                            ):
       
        for datom in donor[dmask[0]]:
            for aatom in acceptor[amask[1]]:

                if(distance(aatom.coord, datom.coord) <= 4.0):
                    return True
    
    return False

def _check_base_stacking(vectors):
    #checks for the presence of base stacking corresponding to the
    #criteria of (Gabb, 1996)
    #   DOI: 10.1016/0263-7855(95)00086-0

    #Check for Base-Base Stacking

    #(i) distance between ring centers <= 4.5 A

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
    
    #(ii) angle between normal vectors <= 23°

    if not ( ( np.arccos(np.dot(vectors[0][3,:], vectors[1][3,:])) )
                <= ( (23*np.pi)/180 )
            ):
            
            return False
    
    #(iii) angle between normalised distance vector and one 
    #   normal vector <= 40°
    
    for vector in vectors:
        for norm_dist_vector in norm_dist_vectors:
            
            if ( np.arccos(np.dot(vector[3,:], norm_dist_vector))
                <= ( (40*np.pi)/180 )
            ):
            
                return True
    
    return False

def _match_base(base, min_atoms):
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
    contains_hydrogens = False

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
        raise UnexpectedStructureWarning("Base Type not supported. Unable to "
                                            "check for basepair")
        return None

    #Check if the structure uses PDBv3 or PDBv2 atom nomenclature

    if( np.sum(np.in1d(std_base[1].atom_name, base.atom_name))
            > np.sum(np.in1d(std_base[0].atom_name, base.atom_name))
    ):
        std_base = std_base[1]
    else:
        std_base = std_base[0]

    #Add the ring centers to the array of vectors to be transformed

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

   
    #print(std_base.coord)
    
    
    vector += trans1
    vector  = np.dot(rot, vector.T).T
    vector += trans2
    
    
    #Normalise the transformed orthogonal base vectors
    
    
    for i in range(1, 4):
        vector[i,:] = vector[i,:]-vector[0,:]
        norm_vector(vector[i,:])
    
    #print(np.dot(vector[1,:], vector[2,:]))

    #If the base is incomplete but contains 3 or more atoms of the 
    #   std_base, transform the complete std_base and use it to
    #   approximate the base.
    
    if(length_difference > 0 and len(fitted) >= min_atoms):
        raise IncompleteStructureWarning("Base is not complete. Attempting "
                                            "to emulate with std_base.")
        ret_base = superimpose_apply(std_base, transformation)
        ret_hpos = std_hpos
        contains_hydrogens = False
    
    #If the base is incomplete and contains less than 3 atoms of the 
    #   std_base throw warning

    elif (length_difference > 0):
        raise IncompleteStructureWarning("Base is smaller than 3 atoms. "
                                            "Unable to check for basepair.")
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
    #TODO: Docstring
    #gets proximate basepairs, where the C1-Sugar-Atoms are within
    # `min_cutoff<=x<=max_cutoff`
    
    c1sugars = array[filter_nucleotides(array) 
                    & _filter_atom_type(array, ["C1'", "C1*"])]
    adj_matrix = CellList(c1sugars, max_cutoff).create_adjacency_matrix(max_cutoff)
    
    basepair_candidates = []
    
    for ix, iy in np.ndindex(adj_matrix.shape):
        if (adj_matrix[ix][iy]):
            candidate = [c1sugars[ix].res_id, c1sugars[ix].chain_id]
            partner = [c1sugars[iy].res_id, c1sugars[iy].chain_id]
            if ((distance(c1sugars[ix].coord, c1sugars[iy].coord) > min_cutoff) 
                 & ((partner + candidate) not in basepair_candidates)):
                
                basepair_candidates.append(candidate + partner)
    
    return basepair_candidates

def _filter_atom_type(array, atom_names):
    # Filter all atoms having the desired `atom_name`.
    return (
        np.in1d(array.atom_name, atom_names) 
        & (array.res_id != -1)
    )

def _filter_residues(array, res_ids, chain_id = None):
    # Filter all atoms having the desired 'residue_id' and 'chain_id'
    if chain_id is None:
        chain_mask =  np.ones(array.array_length(), dtype=bool)
    else:
        chain_mask = np.isin(array.chain_id, chain_id)

    return (
        np.isin(array.res_id, res_ids) 
        & chain_mask
        )
