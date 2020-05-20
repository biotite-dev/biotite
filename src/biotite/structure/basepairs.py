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
import warnings
from .atoms import Atom, array
from .superimpose import superimpose, superimpose_apply
from .filter import filter_nucleotides
from .celllist import CellList
from .hbond import hbond
from .error import IncompleteStructureWarning, UnexpectedStructureWarning
from .util import distance, norm_vector


def  _get_1d_boolean_mask(size, true_ids):
    """
    Get a boolean ndarray of shape=(n,) which can be used as a mask for
    fancy indexing. 
    
    Parameters
    ----------
    size : integer
        Size of the 1-dimensional array.
    true_ids: array_like
        Indices where the boolean mask is `True`.
        
    Returns
    -------
    mask : ndarray, dtype=bool, shape=(n,)
        The boolean mask which is `True` at the specified indices and
        `False` everywhere else.
    """

    mask = np.zeros(size, dtype=bool)
    mask[true_ids] = np.ones(len(true_ids), dtype=bool)
    return mask


def _get_std_adenine():
    """
    Get standard base variables for adenine. 
        
    Returns
    -------
    standard_base : tuple
        Standard coordinates nomenclature of the adenine base, 
        `AtomArray` with nomenclature of PDB File Format V2, `AtomArray`
        with nomenclature of PDB File Format V3
    ring_centers : tuple
        Coordinates of the aromatic ring centers, `ndarray` containing
        the coordinates of the pyrimidine ring center, `ndarray`
        containing the coordinates of the imidazole ring center
    hbond_masks : tuple
        The hydrogen bond donors and acceptors heteroatoms as 'ndarray`
        with dtype=bool, boolean mask for heteroatoms which are bound to
        a hydrogen that can act as a donor, boolean mask for heteroatoms
        that can act as a hydrogen bond acceptor
    """

    atom1 = Atom([-0.68056893, 3.7679946, 0.0], atom_name="C1*", res_name="A")
    atom2 = Atom([0.28598812, 2.6742783, 0.0], atom_name="N9", res_name="A")
    atom3 = Atom([1.6570601, 2.766945, 0.0], atom_name="C8", res_name="A")
    atom4 = Atom([2.2641208, 1.6054324, 0.0], atom_name="N7", res_name="A")
    atom5 = Atom([1.2241601, 0.6849548, 0.0], atom_name="C5", res_name="A")
    atom6 = Atom([1.2053612, -0.71988654, 0.0], atom_name="C6", res_name="A")
    atom7 = Atom([2.3053644, -1.4759803, 0.0], atom_name="N6", res_name="A")
    atom8 = Atom([0.0, -1.3301566, 0.0], atom_name="N1", res_name="A")
    atom9 = Atom([-1.1015016, -0.57166386, 0.0], atom_name="C2", res_name="A")
    atom10 = Atom([-1.2137452, 0.7546673, 0.0], atom_name="N3", res_name="A")
    atom11 = Atom([0.0, 1.3301566, 0.0], atom_name="C4", res_name="A")
    adenine_pdbv2 = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, 
         atom9, atom10, atom11]
                )
    adenine_pdbv3 = adenine_pdbv2.copy()
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

    # Create boolean masks for the AtomArray containing the bases` 
    # heteroatoms which (or the usually attached hydrogens) can act as
    # Hydrogen Bond Donors or Acceptors respectively.
    hbond_donor_mask = _get_1d_boolean_mask(
        adenine_pdbv2.array_length(), [1, 6]
                                        )
    hbond_acceptor_mask = _get_1d_boolean_mask(
        adenine_pdbv2.array_length(), [1, 3, 6, 7, 9]
                                            )

    return (adenine_pdbv2, adenine_pdbv3), \
           (pyrimidine_center, imidazole_center), \
           (hbond_donor_mask, hbond_acceptor_mask)


def _get_std_cytosine():
    """
    Get standard base variables for cytosine. 
        
    Returns
    -------
    standard_base : tuple
        Standard coordinates nomenclature of the cytosine base, 
        `AtomArray` with nomenclature of PDB File Format V2, `AtomArray`
        with nomenclature of PDB File Format V3
    ring_center : tuple
        Coordinates of the aromatic ring centers, `ndarray` containing
        the coordinates of the pyrimidine ring center
    hbond_masks : tuple
        The hydrogen bond donors and acceptors heteroatoms as 'ndarray`
        with dtype=bool, boolean mask for heteroatoms which are bound to
        a hydrogen that can act as a donor, boolean mask for heteroatoms
        that can act as a hydrogen bond acceptor
    """


    atom1 = Atom([-2.4766238, -1.3768258, 0.0], atom_name="C1*", res_name="C")
    atom2 = Atom([-1.1802185, -0.6841519, 0.0], atom_name="N1", res_name="C")
    atom3 = Atom([-1.1802462, 0.71242416, 0.0], atom_name="C2", res_name="C")
    atom4 = Atom([-2.2657278, 1.3121465, 0.0], atom_name="O2", res_name="C")
    atom5 = Atom([0.0, 1.3743725, 0.0], atom_name="N3", res_name="C")
    atom6 = Atom([1.1467924, 0.69068617, 0.0], atom_name="C4", res_name="C")
    atom7 = Atom([2.2880406, 1.3851486, 0.0], atom_name="N4", res_name="C")
    atom8 = Atom([1.1754527, -0.7329672, 0.0], atom_name="C5", res_name="C")
    atom9 = Atom([0.0, -1.3743725, 0.0], atom_name="C6", res_name="C")
    cytosine_pdbv2 = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9]
                    )
    cytosine_pdbv3 = cytosine_pdbv2.copy()
    cytosine_pdbv3.atom_name[[0]] = ["C1'"]

    # Calculate the coordinates of the aromatic ring center.
    pyrimidine_center = np.mean(
        [atom2.coord, atom3.coord, atom5.coord,
         atom6.coord, atom8.coord, atom9.coord], axis=-2
                            )
    
    # Create boolean masks for the AtomArray containing the bases` 
    # heteroatoms which (or the usually attached hydrogens) can act as
    # Hydrogen Bond Donors or Acceptors respectively.
    hbond_donor_mask = _get_1d_boolean_mask(
        cytosine_pdbv2.array_length(), [1, 6]
                                        )
    hbond_acceptor_mask = _get_1d_boolean_mask(
        cytosine_pdbv2.array_length(), [1, 3, 4, 6]
                                            )

    return (cytosine_pdbv2, cytosine_pdbv3), (pyrimidine_center,), \
           (hbond_donor_mask, hbond_acceptor_mask)


def _get_std_guanine():
    """
    Get standard base variables for guanine. 
        
    Returns
    -------
    standard_base : tuple
        Standard coordinates nomenclature of the guanine base, 
        `AtomArray` with nomenclature of PDB File Format V2, `AtomArray`
        with nomenclature of PDB File Format V3
    ring_centers : tuple
        Coordinates of the aromatic ring centers, `ndarray` containing
        the coordinates of the pyrimidine ring center, `ndarray`
        containing the coordinates of the imidazole ring center
    hbond_masks : tuple
        The hydrogen bond donors and acceptors heteroatoms as 'ndarray`
        with dtype=bool, boolean mask for heteroatoms which are bound to
        a hydrogen that can act as a donor, boolean mask for heteroatoms
        that can act as a hydrogen bond acceptor
    """
    
    atom1 = Atom([-0.69979936, 3.731476, 0.0], atom_name="C1*", res_name="G")
    atom2 = Atom([0.2753646, 2.6454268, 0.0], atom_name="N9", res_name="G")
    atom3 = Atom([1.6453435, 2.761283, 0.0], atom_name="C8", res_name="G")
    atom4 = Atom([2.2561362, 1.6078576, 0.0], atom_name="N7", res_name="G")
    atom5 = Atom([1.229222, 0.67279345, 0.0], atom_name="C5", res_name="G")
    atom6 = Atom([1.2752016, -0.7441129, 0.0], atom_name="C6", res_name="G")
    atom7 = Atom([2.2683425, -1.4827579, 0.0], atom_name="O6", res_name="G")
    atom8 = Atom([0.0, -1.2990883, 0.0], atom_name="N1", res_name="G")
    atom9 = Atom([-1.1709266, -0.5812807, 0.0], atom_name="C2", res_name="G")
    atom10 = Atom([-2.3043447, -1.300007, -0.001], atom_name="N2", res_name="G")
    atom11 = Atom([-1.2280219, 0.73974866, 0.001], atom_name="N3", res_name="G")
    atom12 = Atom([0.0, 1.2990883, 0.0], atom_name="C4", res_name="G")
    guanine_pdbv2 = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, 
         atom9, atom10, atom11, atom12]
                )
    guanine_pdbv3 = guanine_pdbv2.copy()
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
    hbond_donor_mask = _get_1d_boolean_mask(
        guanine_pdbv2.array_length(), [1, 7, 9]
                                        )
    hbond_acceptor_mask = _get_1d_boolean_mask(
        guanine_pdbv2.array_length(), [1, 3, 6, 7, 9, 10]
                                            )

    return (guanine_pdbv2, guanine_pdbv3), \
           (pyrimidine_center, imidazole_center), \
           (hbond_donor_mask, hbond_acceptor_mask)


def _get_std_thymine():
    """
    Get standard base variables for thymine. 
        
    Returns
    -------
    standard_base : tuple
        Standard coordinates nomenclature of the thymine base, 
        `AtomArray` with nomenclature of PDB File Format V2, `AtomArray`
        with nomenclature of PDB File Format V3
    ring_center : tuple
        Coordinates of the aromatic ring centers, `ndarray` containing
        the coordinates of the pyrimidine ring center
    hbond_masks : tuple
        The hydrogen bond donors and acceptors heteroatoms as 'ndarray`
        with dtype=bool, boolean mask for heteroatoms which are bound to
        a hydrogen that can act as a donor, boolean mask for heteroatoms
        that can act as a hydrogen bond acceptor
    """
    atom1 = Atom([-2.4745166, -1.3747915, -0.0], atom_name="C1*", res_name="T")
    atom2 = Atom([-1.196032, -0.64842904, -0.0], atom_name="N1", res_name="T")
    atom3 = Atom([-1.2327007, 0.72763944, -0.0], atom_name="C2", res_name="T")
    atom4 = Atom([-2.2726667, 1.3649775, -0.0], atom_name="O2", res_name="T")
    atom5 = Atom([0.0, 1.3320639, -0.0], atom_name="N3", res_name="T")
    atom6 = Atom([1.2347531, 0.7117828, -0.0], atom_name="C4", res_name="T")
    atom7 = Atom([2.2597313, 1.3879516, -0.0], atom_name="O4", res_name="T")
    atom8 = Atom([1.1979554, -0.73309445, -0.0], atom_name="C5", res_name="T")
    atom9 = Atom([2.4866693, -1.4926641, -0.001], atom_name="C5M", res_name="T")
    atom10 = Atom([0.0, -1.3320639, -0.0], atom_name="C6", res_name="T")
    thymine_pdbv2 = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9, atom10]
                )
    thymine_pdbv3 = thymine_pdbv2.copy()
    thymine_pdbv3.atom_name[[0, 8]] = ["C1'", "C7"]

    # Calculate the coordinates of the aromatic ring center.
    pyrimidine_center = np.mean(
        [atom2.coord, atom3.coord, atom5.coord,
         atom6.coord, atom8.coord, atom10.coord], axis=-2
                            )

    # Create boolean masks for the AtomArray containing the bases` 
    # heteroatoms which (or the usually attached hydrogens) can act as
    # Hydrogen Bond Donors or Acceptors respectively.
    hbond_donor_mask = _get_1d_boolean_mask(
        thymine_pdbv2.array_length(), [1, 4]
                                        )
    hbond_acceptor_mask = _get_1d_boolean_mask(
        thymine_pdbv2.array_length(), [1, 3, 4, 6]
                                            )
      
    return (thymine_pdbv2, thymine_pdbv3), (pyrimidine_center,), \
           (hbond_donor_mask, hbond_acceptor_mask)


def _get_std_uracil():
    """
    Get standard base variables for uracil. 
        
    Returns
    -------
    standard_base : tuple
        Standard coordinates nomenclature of the uracil base, 
        `AtomArray` with nomenclature of PDB File Format V2, `AtomArray`
        with nomenclature of PDB File Format V3
    ring_center : tuple
        Coordinates of the aromatic ring centers, `ndarray` containing
        the coordinates of the pyrimidine ring center
    hbond_masks : tuple
        The hydrogen bond donors and acceptors heteroatoms as 'ndarray`
        with dtype=bool, boolean mask for heteroatoms which are bound to
        a hydrogen that can act as a donor, boolean mask for heteroatoms
        that can act as a hydrogen bond acceptor
    """

    atom1 = Atom([-2.4749846, -1.378846, -0.0], atom_name="C1*", res_name="U")
    atom2 = Atom([-1.195587, -0.6540935, -0.0], atom_name="N1", res_name="U")
    atom3 = Atom([-1.2301071, 0.7259982, -0.0], atom_name="C2", res_name="U")
    atom4 = Atom([-2.270681, 1.3607708, -0.0], atom_name="O2", res_name="U")
    atom5 = Atom([0.0, 1.3352546, -0.0], atom_name="N3", res_name="U")
    atom6 = Atom([1.233289, 0.7165072, -0.0], atom_name="C4", res_name="U")
    atom7 = Atom([2.2563882, 1.4037365, 0.001], atom_name="O4", res_name="U")
    atom8 = Atom([1.184195, -0.7131495, -0.0], atom_name="C5", res_name="U")
    atom9 = Atom([0.0, -1.3352546, -0.0], atom_name="C6", res_name="U")
    uracil_pdbv2 = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9]
                )
    uracil_pdbv3 = uracil_pdbv2.copy()
    uracil_pdbv3.atom_name[[0]] = ["C1'"]

    # Calculate the coordinates of the aromatic ring center.
    pyrimidine_center = np.mean(
        [atom2.coord, atom3.coord, atom5.coord,
         atom6.coord, atom8.coord, atom9.coord], axis=-2
                            )

    # Create boolean masks for the AtomArray containing the bases` 
    # heteroatoms which (or the usually attached hydrogens) can act as
    # Hydrogen Bond Donors or Acceptors respectively.
    hbond_donor_mask = _get_1d_boolean_mask(
        uracil_pdbv2.array_length(), [1, 4]
                                        )
    hbond_acceptor_mask = _get_1d_boolean_mask(
        uracil_pdbv2.array_length(), [1, 3, 4, 6]
                                            )

    return (uracil_pdbv2, uracil_pdbv3), (pyrimidine_center,), \
           (hbond_donor_mask, hbond_acceptor_mask)


_std_adenine, _std_adenine_ring_centers, \
        _std_adenine_hbond_masks = _get_std_adenine()
_std_cytosine, _std_cytosine_ring_centers, \
        _std_cytosine_hbond_masks = _get_std_cytosine()
_std_guanine, _std_guanine_ring_centers, \
        _std_guanine_hbond_masks = _get_std_guanine()  
_std_thymine, _std_thymine_ring_centers, \
        _std_thymine_hbond_masks = _get_std_thymine()
_std_uracil, _std_uracil_ring_centers, \
        _std_uracil_hbond_masks = _get_std_uracil()

_adenine_containing_nucleotides = ["A", "DA"]
_thymine_containing_nucleotides = ["T", "DT"]
_cytosine_containing_nucleotides = ["C", "DC"]
_guanine_containing_nucleotides = ["G", "DG"]
_uracil_containing_nucleotides = ["U", "DU"]


def get_basepairs(atom_array, min_atoms_per_base = 3):
    """
    Use DSSR criteria to find the basepairs in an `Atom Array`.
    
    Parameters
    ----------
    atom_array : AtomArray
        The `AtomArray` to find basepairs in.
    min_atoms_per_base: integer
        Indices where the boolean mask is `True`.
        
    Returns
    -------
    mask : ndarray, dtype=bool, shape=(n,)
        The boolean mask which is `True` at the specified indices and
        `False` everywhere else.
    """

    basepair_candidates = _get_proximate_basepair_candidates(atom_array)
    basepairs = []

    for base1_index, base2_index in basepair_candidates:
        base1 = atom_array[_filter_residues(atom_array, base1_index)]
        base2 = atom_array[_filter_residues(atom_array, base2_index)]
        if _check_dssr_criteria([base1, base2], min_atoms_per_base):
            basepairs.append((base1_index, base2_index))
    
    return basepairs


def _check_dssr_criteria(basepair, min_atoms_per_base):
    transformed_bases = [None] * 2
    hbond_masks = [None] * 2
    # A list containing one NumPy array for each base with transformed
    # vectors from the standard base reference frame to the structures
    # coordinates. The layout is as follows:
    #
    # [Origin coordinates]
    # [Orthonormal base vectors] * 3 in the order x, y, z
    # [Aromatic Ring Center coordinates]
    transformed_std_vectors = [None] * 2
    contains_hydrogens = np.zeros(2, dtype=bool)
    is_purine = np.zeros(2, dtype=bool)

    # Generate the data necessary for analysis for each base.
    for i in range(2):
        base_tuple = _match_base(basepair[i], min_atoms_per_base)
        
        if(base_tuple is None):
            return False
        
        transformed_bases[i], hbond_masks[i], contains_hydrogens[i], \
            transformed_std_vectors[i], is_purine[i] = base_tuple

    
    # Criterion 1: Distance between orgins <= 15 Å
    if not (distance(transformed_std_vectors[0][0,:],
            transformed_std_vectors[1][0,:]) <= 15):
        return False
    
    # Criterion 2: Vertical seperation <= 2.5 Å
    #
    # Calculate the angle between normal vectors of the bases
    normal_vector_angle = np.arccos(np.dot(transformed_std_vectors[0][3,:],
                                           transformed_std_vectors[1][3,:]))
    # Calculate the orthonormal vector to the normal vector of the bases
    rotation_axis = np.cross(transformed_std_vectors[0][3,:],
                                 transformed_std_vectors[1][3,:])
    norm_vector(rotation_axis)
    # Rotate the base normal vectors by ± half the angle between the two
    # vectors
    rotated_normal_vector_0 = np.dot(
        _get_rotation_matrix(rotation_axis, normal_vector_angle/2),
        transformed_std_vectors[0][3,:]
                                )
    rotated_normal_vector_1 = np.dot(
        _get_rotation_matrix(rotation_axis, ((-1)*normal_vector_angle)/2),
        transformed_std_vectors[1][3,:]
                                )
    # Average and normalize the rotated vectors
    z_rot_average = (rotated_normal_vector_0 + rotated_normal_vector_1)/2
    norm_vector(z_rot_average)
    # Calculate the vector between the two origins    
    origin_vector = transformed_std_vectors[1][0,:] \
                    - transformed_std_vectors[0][0,:]
    # The angle between the averaged rotated normal vectors and the 
    # vector between the two origins is the vertical seperation
    if not abs(int(np.dot(origin_vector, z_rot_average))) <= 2.5:
        return False
    
    # Criterion 3: Angle between normal vectors <= 65°
    if not (np.arccos(np.dot(transformed_std_vectors[0][3,:],
                              transformed_std_vectors[1][3,:])
                    ) <=
            ((65*np.pi)/180)
        ):
        return False
   
    # Criterion 4: Absence of stacking
    if _check_base_stacking(transformed_std_vectors):
        return False
    
    # Criterion 5: Presence of at least on hydrogen bond
    # Check if both bases came with hydrogens.
    if (np.all(contains_hydrogens)):
        # For Structures that contain hydrogens, check for their 
        # presence directly.
        if(len(hbond(transformed_bases[0] + transformed_bases[1],
                     np.ones_like(
                         transformed_bases[0] + transformed_bases[1],
                         dtype=bool
                                ), 
                     np.ones_like(
                         transformed_bases[0] + transformed_bases[1],
                         dtype=bool
                                )
                    )
            ) == 0):
            return False           
    elif not _check_hbonds(transformed_bases, hbond_masks):
        # if the structure does not contain hydrogens, check for
        # plausibility of hydrogen bonds between heteroatoms
        return False
    
    return True


def _check_hbonds(bases, hbond_masks):
    for donor_base, hbond_donor_mask, acceptor_base, hbond_acceptor_mask in \
        zip(bases, hbond_masks, reversed(bases), reversed(hbond_masks)):
        for donor_atom in donor_base[hbond_donor_mask[0]]:
            for acceptor_atom in acceptor_base[hbond_acceptor_mask[1]]:
                if(distance(acceptor_atom.coord, donor_atom.coord) <= 4.0):
                    return True
    return False


def _check_base_stacking(transformed_vectors):
    # Check for the presence of base stacking corresponding to the
    # criteria of (Gabb, 1996): DOI: 10.1016/0263-7855(95)00086-0

    # Contains the normalized distance vectors between ring centers less
    # than 4.5 Å apart.
    normalized_distance_vectors = []

    # Criterion 1: Distance between aromatic ring centers <= 4.5 Å
    wrongdistance = True
    for ring_center1 in transformed_vectors[0][4:][:]:
        for ring_center2 in transformed_vectors[1][4:][:]:
            if (distance(ring_center1, ring_center2) <= 4.5):
                wrongdistance = False
                normalized_distance_vectors.append(ring_center2 - ring_center1)
                norm_vector(normalized_distance_vectors[-1]) 
    if(wrongdistance == True):
        return False
    
    # Criterion 2: Angle between normal vectors <= 23°
    if not ((np.arccos(np.dot(transformed_vectors[0][3,:],
                              transformed_vectors[1][3,:])))
            <= ((23*np.pi)/180)):
        return False
    
    # Criterion 3: Angle between normalized distance vector and one 
    # normal vector <= 40°
    for vector in transformed_vectors:
        for normalized_dist_vector in normalized_distance_vectors:    
            if (np.arccos(np.dot(vector[3,:], normalized_dist_vector))
                <= ((40*np.pi)/180)):
                return True
    
    return False


def _match_base(base, min_atoms_per_base):
    # Matches a nucleotide to a standard base
    # Returns: 
    # ret_base : The base or if the base atoms are incomplete a
    #               superimposed standard base
    # ret_hpos : A list of size 2 containing boolean masks. 
    #               Pos 0 contains the het_atoms that act as H-Donors
    #               Pos 1 contains the het_atoms that act as H-Acceptors
    # contains_hydrogens : A boolean; if True the base contains H-Atoms
    # vectors : A set of std_vectors (Origin, Orthonormal-Base-Vectors, 
    #               Ring-Centers) transformed onto the
    #               nucleotides coordinates   

    return_hbond_masks = [None] * 2
    contains_hydrogens = False

    vectors = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                             [0, 0, 1]], np.float)

    # Check base type and match standard base.
    if(base[0].res_name in _adenine_containing_nucleotides):
        std_base = _std_adenine
        std_ring_centers = _std_adenine_ring_centers
        std_hbond_masks = _std_adenine_hbond_masks
        is_purine = True
    elif(base[0].res_name in _thymine_containing_nucleotides):
        std_base = _std_thymine
        std_ring_centers = _std_thymine_ring_centers
        std_hbond_masks = _std_thymine_hbond_masks
        is_purine = False
    elif(base[0].res_name in _cytosine_containing_nucleotides):
        std_base = _std_cytosine
        std_ring_centers = _std_cytosine_ring_centers
        std_hbond_masks = _std_cytosine_hbond_masks
        is_purine = False
    elif(base[0].res_name in _guanine_containing_nucleotides):
        std_base = _std_guanine
        std_ring_centers = _std_guanine_ring_centers
        std_hbond_masks = _std_guanine_hbond_masks
        is_purine = True
    elif(base[0].res_name in _uracil_containing_nucleotides):
        std_base = _std_uracil
        std_ring_centers = _std_uracil_ring_centers
        std_hbond_masks = _std_uracil_hbond_masks
        is_purine = False 
    else:
        warnings.warn("Base Type not supported. Unable to check for basepair",
                      UnexpectedStructureWarning)
        return None

    # Check if the structure uses PDBv3 or PDBv2 atom nomenclature.
    if (np.sum(np.in1d(std_base[1].atom_name, base.atom_name))
        > np.sum(np.in1d(std_base[0].atom_name, base.atom_name))):
        std_base = std_base[1]
    else:
        std_base = std_base[0]

    # Add the ring centers to the array of vectors to be transformed.
    vectors = np.vstack((vectors, std_ring_centers))
    
    # Match the selected std_base to the base.
    fitted, transformation = superimpose(
                        base[np.in1d(base.atom_name, std_base.atom_name)],
                        std_base[np.in1d(std_base.atom_name, base.atom_name)]
                                        )
    # Transform the vectors
    trans1, rot, trans2 = transformation
    vectors += trans1
    vectors  = np.dot(rot, vectors.T).T
    vectors += trans2   
    # Normalize the transformed orthogonal base vectors   
    for i in range(1, 4):
        vectors[i,:] = vectors[i,:]-vectors[0,:]
        norm_vector(vectors[i,:])

    # Investigate the completeness of the base:
    # 
    # A length difference of zero means the base contains all atoms of
    # the std_base          
    length_difference = len(std_base) - len(fitted)
    
    if(length_difference > 0 and len(fitted) >= min_atoms_per_base):
        # If the base is incomplete but contains 3 or more atoms of the 
        # std_base, transform the complete std_base and use it to
        # approximate the base.
        warnings.warn("Base is not complete. Attempting to emulate with "
                      "std_base.", IncompleteStructureWarning)
        return_base = superimpose_apply(std_base, transformation)
        return_hbond_masks = std_hbond_masks
        contains_hydrogens = False
    elif (length_difference > 0):
        # If the base is incomplete and contains less than 3 atoms of 
        # the std_base throw warning
        warnings.warn("Base is smaller than 3 atoms. Unable to check for "
                     "basepair.", IncompleteStructureWarning)
        return None
    else:
        # If the base is complete use the base for further calculations.
        #
        # Generate a boolean mask containing only the base atoms and
        # their hydrogens(if available), disregarding the sugar atoms
        # and the phosphate backbone.
        base_atom_mask = np.ones(len(base), dtype=bool)
        for i in range(len(base)):
            if (
                ("'" in base[i].atom_name) or ("*" in base[i].atom_name) or
                ((base[i].atom_name not in std_base.atom_name) and
                 (base[i].element != "H"))
            ):
                base_atom_mask[i] = False
        
        # Create boolean masks for the AtomArray containing the bases` 
        # heteroatoms, which (or the usually attached hydrogens) can act 
        # as Hydrogen Bond Donors or Acceptors respectively, using the
        # std_base as a template.
        for i in range(2):
            return_hbond_masks[i] = _filter_atom_type(base[base_atom_mask], 
                                std_base[std_hbond_masks[i]].atom_name)

        # Check if the base contains Hydrogens.
        if ("H" in base.element[base_atom_mask]):
            contains_hydrogens = True
            return_base = base[base_atom_mask]          
        else:
            return_base = base[base_atom_mask]
        
    return return_base, return_hbond_masks, \
           contains_hydrogens, vectors, is_purine


def _get_proximate_basepair_candidates(atom_array, max_cutoff = 15,
                                       min_cutoff = 9):
    # Get a boolean mask for the C1 sugar atoms
    c1mask = (filter_nucleotides(atom_array) 
              & _filter_atom_type(atom_array, ["C1'", "C1*"]))
    # Get the indices of the C1 atoms that are within the maximum cutoff
    # of each other
    indices = CellList( 
        atom_array, max_cutoff, selection=c1mask
                    ).get_atoms(atom_array.coord[c1mask], max_cutoff)
    
    # Loop through the indices of potential partners
    basepair_candidates = []
    for candidate, partners in zip(np.argwhere(c1mask), indices):
        for partner in partners:
            if partner == -1:
                break
            # Check if the basepair candidates have a distance which is
            # greater than the minimum cutoff
            if(distance(
                atom_array[candidate].coord, atom_array[partner].coord
                        ) > min_cutoff
            ): 
                # Find the indices of the first atom of the residues
                candidate_res_start = np.where(
                    (atom_array.res_id == atom_array[candidate].res_id)
                    & (atom_array.chain_id == atom_array[candidate].chain_id)
                                            )[0][0]
                partner_res_start = np.where(
                    (atom_array.res_id == atom_array[partner].res_id)
                    & (atom_array.chain_id == atom_array[partner].chain_id)
                                )[0][0]
                # If the countperpart of the basepair candidates is not
                # already in the output list, append to the output
                if (partner_res_start, candidate_res_start) \
                    not in basepair_candidates:
                    basepair_candidates.append(
                        (candidate_res_start, partner_res_start)
                                            )
    return basepair_candidates


def _filter_atom_type(atom_array, atom_names):
    # Filter all atoms having the desired `atom_name`.
    return (np.isin(atom_array.atom_name, atom_names)
            & (atom_array.res_id != -1))


def _filter_residues(atom_array, index):
    return (np.isin(atom_array.res_id, atom_array[int(index)].res_id)
            & np.isin(atom_array.chain_id, atom_array[int(index)].chain_id))
        
def _get_rotation_matrix(axis, angle):
    rot_matrix = np.zeros((3,3), dtype=float)
    n1, n2, n3 = axis
    normal_cos = np.cos(angle)
    inverse_cos = 1 - normal_cos

    rot_matrix[0,0] = ((n1*n1)*inverse_cos) + normal_cos
    rot_matrix[0,1] = ((n1*n2)*inverse_cos) - (n3*np.sin(angle))
    rot_matrix[0,2] = ((n1*n3)*inverse_cos) + (n2*np.sin(angle))
    rot_matrix[1,0] = ((n2*n1)*inverse_cos) + (n3*np.sin(angle))
    rot_matrix[1,1] = ((n2*n2)*inverse_cos) + normal_cos
    rot_matrix[1,2] = ((n2*n3)*inverse_cos) - (n1*np.sin(angle))
    rot_matrix[2,0] = ((n3*n1)*inverse_cos) - (n2*np.sin(angle))
    rot_matrix[2,1] = ((n3*n2)*inverse_cos) + (n1*np.sin(angle))
    rot_matrix[2,2] = ((n3*n3)*inverse_cos) + normal_cos

    return rot_matrix