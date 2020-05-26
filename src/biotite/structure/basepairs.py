# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for basepair identification.
"""

__name__ = "biotite.structure"
__author__ = "Tom David Müller"
__all__ = ["base_pairs"]

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
    Get a boolean mask for fancy indexing. 
    
    Parameters
    ----------
    size : integer
        Size of the 1-dimensional array.
    true_ids: array_like
        Indices where the boolean mask is ``True``.
        
    Returns
    -------
    mask : ndarray, dtype=bool, shape=(n,)
        The boolean mask which is ``True`` at the specified indices and
        ``False`` everywhere else.
    """
    mask = np.zeros(size, dtype=bool)
    mask[true_ids] = np.ones(len(true_ids), dtype=bool)
    return mask


def _get_std_adenine():
    """
    Get standard base variables for adenine. 
        
    Returns
    -------
    standard_base : tuple (AtomArray, AtomArray)
        Standard coordinates nomenclature of the adenine base, 
        :class:`AtomArray` with nomenclature of PDB File Format V2, 
        :class:`AtomArray` with nomenclature of PDB File Format V3
    coordinates : tuple (ndarray, ndarray, ndarray, dtype=float)
        :class:`ndarray` containing the center according to the SCHNaP-
        paper referenced in the function ``base_pairs``, 
        :class:`ndarray` containing the coordinates of the pyrimidine
        ring center, :class:`ndarray` containing the coordinates of the 
        imidazole ring center
    hbond_masks : tuple (ndarray, ndarray, dtype=bool)
        The hydrogen bond donor and acceptor heteroatoms as 
        :class:`ndarray` with ``dtype=bool``, boolean mask for
        heteroatoms which are bound to a hydrogen that can act as a 
        donor, boolean mask for heteroatoms that can act as a hydrogen
        bond acceptor 
    """
    atom1 =  Atom([-2.479, 5.346, 0.000], atom_name="C1*", res_name="A")
    atom2 =  Atom([-1.291, 4.498, 0.000], atom_name="N9",  res_name="A")
    atom3 =  Atom([0.024, 4.897, 0.000],  atom_name="C8",  res_name="A")
    atom4 =  Atom([0.877, 3.902, 0.000],  atom_name="N7",  res_name="A")
    atom5 =  Atom([0.071, 2.771, 0.000],  atom_name="C5",  res_name="A")
    atom6 =  Atom([0.369, 1.398, 0.000],  atom_name="C6",  res_name="A")
    atom7 =  Atom([1.611, 0.909, 0.000],  atom_name="N6",  res_name="A")
    atom8 =  Atom([-0.668, 0.532, 0.000], atom_name="N1",  res_name="A")
    atom9 =  Atom([-1.912, 1.023, 0.000], atom_name="C2",  res_name="A")
    atom10 = Atom([-2.320, 2.290, 0.000], atom_name="N3",  res_name="A")
    atom11 = Atom([-1.267, 3.124, 0.000], atom_name="C4",  res_name="A")
    adenine_pdbv2 = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, 
         atom9, atom10, atom11]
    )
    adenine_pdbv3 = adenine_pdbv2.copy()
    adenine_pdbv3.atom_name[[0]] = ["C1'"]

    # Get the midpoint between the N1 and C4 atoms
    midpoint = np.mean([atom8.coord, atom11.coord], axis=-2)
    # Calculate the coordinates of the aromatic ring centers
    pyrimidine_center = np.mean(
        [atom5.coord, atom6.coord, atom8.coord, 
         atom9.coord, atom10.coord, atom11.coord], axis=-2
    )
    imidazole_center = np.mean(
        [atom2.coord, atom3.coord, atom4.coord,
         atom5.coord, atom11.coord], axis=-2
    )

    # Create boolean masks for the AtomArray containing the bases` 
    # heteroatoms (or the usually attached hydrogens) which can act as
    # Hydrogen Bond Donors or Acceptors respectively.
    hbond_donor_mask = _get_1d_boolean_mask(
        adenine_pdbv2.array_length(), [1, 6]
    )
    hbond_acceptor_mask = _get_1d_boolean_mask(
        adenine_pdbv2.array_length(), [1, 3, 6, 7, 9]
    )

    return (adenine_pdbv2, adenine_pdbv3), \
           (midpoint, pyrimidine_center, imidazole_center), \
           (hbond_donor_mask, hbond_acceptor_mask)


def _get_std_cytosine():
    """
    Get standard base variables for cytosine. 
        
    Returns
    -------
   standard_base : tuple (AtomArray, AtomArray)
        Standard coordinates nomenclature of the adenine base, 
        :class:`AtomArray` with nomenclature of PDB File Format V2, 
        :class:`AtomArray` with nomenclature of PDB File Format V3
    coordinates : tuple (ndarray, ndarray, dtype=float)
        :class:`ndarray` containing the center according to the SCHNaP-
        paper referenced in the function ``base_pairs``, 
        :class:`ndarray` containing the coordinates of the pyrimidine
        ring center
    hbond_masks : tuple (ndarray, ndarray, dtype=bool)
        The hydrogen bond donors and acceptors heteroatoms as 
        :class:`ndarray` with ``dtype=bool``, boolean mask for
        heteroatoms which are bound to a hydrogen that can act as a 
        donor, boolean mask for heteroatoms that can act as a hydrogen
        bond acceptor
    """
    atom1 = Atom([-2.477, 5.402, 0.000], atom_name="C1*", res_name="C")
    atom2 = Atom([-1.285, 4.542, 0.000], atom_name="N1",  res_name="C")
    atom3 = Atom([-1.472, 3.158, 0.000], atom_name="C2",  res_name="C")
    atom4 = Atom([-2.628, 2.709, 0.000], atom_name="O2",  res_name="C")
    atom5 = Atom([-0.391, 2.344, 0.000], atom_name="N3",  res_name="C")
    atom6 = Atom([0.837, 2.868, 0.000],  atom_name="C4",  res_name="C")
    atom7 = Atom([1.875, 2.027, 0.000],  atom_name="N4",  res_name="C")
    atom8 = Atom([1.056, 4.275, 0.000],  atom_name="C5",  res_name="C")
    atom9 = Atom([-0.023, 5.068, 0.000], atom_name="C6",  res_name="C")
    cytosine_pdbv2 = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9]
    )
    cytosine_pdbv3 = cytosine_pdbv2.copy()
    cytosine_pdbv3.atom_name[[0]] = ["C1'"]

    # Get the midpoint between the N3 and C6 atoms
    midpoint = np.mean([atom5.coord, atom9.coord], axis=-2)
    # Calculate the coordinates of the aromatic ring center
    pyrimidine_center = np.mean(
        [atom2.coord, atom3.coord, atom5.coord,
         atom6.coord, atom8.coord, atom9.coord], axis=-2
    )
    
    # Create boolean masks for the AtomArray containing the bases` 
    # heteroatoms (or the usually attached hydrogens) which can act as
    # Hydrogen Bond Donors or Acceptors respectively
    hbond_donor_mask = _get_1d_boolean_mask(
        cytosine_pdbv2.array_length(), [1, 6]
    )
    hbond_acceptor_mask = _get_1d_boolean_mask(
        cytosine_pdbv2.array_length(), [1, 3, 4, 6]
    )

    return (cytosine_pdbv2, cytosine_pdbv3), (midpoint, pyrimidine_center), \
           (hbond_donor_mask, hbond_acceptor_mask)


def _get_std_guanine():
    """
    Get standard base variables for guanine. 
        
    Returns
    -------
    standard_base : tuple (AtomArray, AtomArray)
        Standard coordinates nomenclature of the adenine base, 
        :class:`AtomArray` with nomenclature of PDB File Format V2, 
        :class:`AtomArray` with nomenclature of PDB File Format V3
    coordinates : tuple (ndarray, ndarray, ndarray, dtype=float)
        :class:`ndarray` containing the center according to the SCHNaP-
        paper referenced in the function ''base_pairs'',
        :class:`ndarray` containing the coordinates of the pyrimidine 
        ring center, :class:`ndarray` containing the coordinates of the
        imidazole ring center
    hbond_masks : tuple (ndarray, ndarray, dtype=bool)
        The hydrogen bond donors and acceptors heteroatoms as 
        :class:`ndarray` with ``dtype=bool``, boolean mask for
        heteroatoms which are bound to a hydrogen that can act as a 
        donor, boolean mask for heteroatoms that can act as a hydrogen
        bond acceptor
    """
    atom1 =  Atom([-2.477, 5.399, 0.000],  atom_name="C1*", res_name="G")
    atom2 =  Atom([-1.289, 4.551, 0.000],  atom_name="N9",  res_name="G")
    atom3 =  Atom([0.023, 4.962, 0.000],   atom_name="C8",  res_name="G")
    atom4 =  Atom([0.870, 3.969, 0.000],   atom_name="N7",  res_name="G")
    atom5 =  Atom([0.071, 2.833, 0.000],   atom_name="C5",  res_name="G")
    atom6 =  Atom([0.424, 1.460, 0.000],   atom_name="C6",  res_name="G")
    atom7 =  Atom([1.554, 0.955, 0.000],   atom_name="O6",  res_name="G")
    atom8 =  Atom([-0.700, 0.641, 0.000],  atom_name="N1",  res_name="G")
    atom9 =  Atom([-1.999, 1.087, 0.000],  atom_name="C2",  res_name="G")
    atom10 = Atom([-2.949, 0.139, -0.001], atom_name="N2",  res_name="G")
    atom11 = Atom([-2.342, 2.364, 0.001],  atom_name="N3",  res_name="G")
    atom12 = Atom([-1.265, 3.177, 0.000],  atom_name="C4",  res_name="G")
    guanine_pdbv2 = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, 
         atom9, atom10, atom11, atom12]
    )
    guanine_pdbv3 = guanine_pdbv2.copy()
    guanine_pdbv3.atom_name[[0]] = ["C1'"]

    # Get the midpoint between the N1 and C4 atoms
    midpoint = np.mean([atom8.coord, atom12.coord], axis=-2)
    # Calculate the coordinates of the aromatic ring centers
    pyrimidine_center = np.mean(
        [atom5.coord, atom6.coord, atom8.coord,
         atom9.coord, atom11.coord, atom12.coord], axis=-2
    )
    imidazole_center = np.mean(
        [atom2.coord, atom3.coord, atom4.coord,
         atom5.coord, atom12.coord], axis=-2
    )

    # Create boolean masks for the AtomArray containing the bases` 
    # heteroatoms (or the usually attached hydrogens) which can act as
    # Hydrogen Bond Donors or Acceptors respectively
    hbond_donor_mask = _get_1d_boolean_mask(
        guanine_pdbv2.array_length(), [1, 7, 9]
    )
    hbond_acceptor_mask = _get_1d_boolean_mask(
        guanine_pdbv2.array_length(), [1, 3, 6, 7, 9, 10]
    )

    return (guanine_pdbv2, guanine_pdbv3), \
           (midpoint, pyrimidine_center, imidazole_center), \
           (hbond_donor_mask, hbond_acceptor_mask)


def _get_std_thymine():
    """
    Get standard base variables for thymine. 
        
    Returns
    -------
    standard_base : tuple (AtomArray, AtomArray)
        Standard coordinates nomenclature of the adenine base, 
        :class:`AtomArray` with nomenclature of PDB File Format V2, 
        :class:`AtomArray` with nomenclature of PDB File Format V3
    coordinates : tuple (ndarray, ndarray, dtype=float)
        :class:`ndarray` containing the center according to the SCHNaP-
        paper referenced in the function ``base_pairs``, 
        :class:`ndarray` containing the coordinates of the pyrimidine 
        ring center
    hbond_masks : tuple (ndarray, ndarray, dtype=bool)
        The hydrogen bond donors and acceptors heteroatoms as 
        :class:`ndarray` with ``dtype=bool``, boolean mask for
        heteroatoms which are bound to a hydrogen that can act as a 
        donor, boolean mask for heteroatoms that can act as a hydrogen
        bond acceptor
    """
    atom1 =  Atom([-2.481, 5.354, 0.000], atom_name="C1*", res_name="T")
    atom2 =  Atom([-1.284, 4.500, 0.000], atom_name="N1",  res_name="T")
    atom3 =  Atom([-1.462, 3.135, 0.000], atom_name="C2",  res_name="T")
    atom4 =  Atom([-2.562, 2.608, 0.000], atom_name="O2",  res_name="T")
    atom5 =  Atom([-0.298, 2.407, 0.000], atom_name="N3",  res_name="T")
    atom6 =  Atom([0.994, 2.897, 0.000],  atom_name="C4",  res_name="T")
    atom7 =  Atom([1.944, 2.119, 0.000],  atom_name="O4",  res_name="T")
    atom8 =  Atom([1.106, 4.338, 0.000],  atom_name="C5",  res_name="T")
    atom9 =  Atom([2.466, 4.961, 0.001],  atom_name="C5M", res_name="T")
    atom10 = Atom([-0.024, 5.057, 0.000], atom_name="C6",  res_name="T")
    thymine_pdbv2 = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9, atom10]
    )
    thymine_pdbv3 = thymine_pdbv2.copy()
    thymine_pdbv3.atom_name[[0, 8]] = ["C1'", "C7"]

    # Get the midpoint between the N3 and C6 atoms
    midpoint = np.mean([atom5.coord, atom10.coord], axis=-2)
    # Calculate the coordinates of the aromatic ring center
    pyrimidine_center = np.mean(
        [atom2.coord, atom3.coord, atom5.coord,
         atom6.coord, atom8.coord, atom10.coord], axis=-2
    )

    # Create boolean masks for the AtomArray containing the bases` 
    # heteroatoms(or the usually attached hydrogens) which can act as
    # Hydrogen Bond Donors or Acceptors respectively
    hbond_donor_mask = _get_1d_boolean_mask(
        thymine_pdbv2.array_length(), [1, 4]
    )
    hbond_acceptor_mask = _get_1d_boolean_mask(
        thymine_pdbv2.array_length(), [1, 3, 4, 6]
    )
      
    return (thymine_pdbv2, thymine_pdbv3), (midpoint, pyrimidine_center), \
           (hbond_donor_mask, hbond_acceptor_mask)


def _get_std_uracil():
    """
    Get standard base variables for uracil. 
        
    Returns
    -------
    standard_base : tuple (AtomArray, AtomArray)
        Standard coordinates nomenclature of the adenine base, 
        :class:`AtomArray` with nomenclature of PDB File Format V2, 
        :class:`AtomArray` with nomenclature of PDB File Format V3
    coordinates : tuple (ndarray, ndarray, dtype=float)
        :class:`ndarray` containing the center according to the SCHNaP-
        paper referenced in the function ``base_pairs``, 
        :class:`ndarray` containing the coordinates of the pyrimidine
        ring center
    hbond_masks : tuple (ndarray, ndarray, dtype=bool)
        The hydrogen bond donors and acceptors heteroatoms as 
        :class:`ndarray` with ``dtype=bool``, boolean mask for
        heteroatoms which are bound to a hydrogen that can act as a 
        donor, boolean mask for heteroatoms that can act as a hydrogen
        bond acceptor
    """
    atom1 = Atom([-2.481, 5.354, 0.000], atom_name="C1*", res_name="U")
    atom2 = Atom([-1.284, 4.500, 0.000], atom_name="N1",  res_name="U")
    atom3 = Atom([-1.462, 3.131, 0.000], atom_name="C2",  res_name="U")
    atom4 = Atom([-2.563, 2.608, 0.000], atom_name="O2",  res_name="U")
    atom5 = Atom([-0.302, 2.397, 0.000], atom_name="N3",  res_name="U")
    atom6 = Atom([0.989, 2.884, 0.000],  atom_name="C4",  res_name="U")
    atom7 = Atom([1.935, 2.094, -0.001], atom_name="O4",  res_name="U")
    atom8 = Atom([1.089, 4.311, 0.000],  atom_name="C5",  res_name="U")
    atom9 = Atom([-0.024, 5.053, 0.000], atom_name="C6",  res_name="U")
    uracil_pdbv2 = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9]
    )
    uracil_pdbv3 = uracil_pdbv2.copy()
    uracil_pdbv3.atom_name[[0]] = ["C1'"]

    # Get the midpoint between the N3 and C6 atoms
    midpoint = np.mean([atom5.coord, atom9.coord], axis=-2)
    # Calculate the coordinates of the aromatic ring center
    pyrimidine_center = np.mean(
        [atom2.coord, atom3.coord, atom5.coord,
         atom6.coord, atom8.coord, atom9.coord], axis=-2
    )

    # Create boolean masks for the AtomArray containing the bases` 
    # heteroatoms (or the usually attached hydrogens) which can act as
    # Hydrogen Bond Donors or Acceptors respectively
    hbond_donor_mask = _get_1d_boolean_mask(
        uracil_pdbv2.array_length(), [1, 4]
    )
    hbond_acceptor_mask = _get_1d_boolean_mask(
        uracil_pdbv2.array_length(), [1, 3, 4, 6]
    )

    return (uracil_pdbv2, uracil_pdbv3), (midpoint, pyrimidine_center), \
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


def base_pairs(atom_array, min_atoms_per_base = 3, unique = True):
    """
    Use DSSR criteria [1]_ to find the basepairs in an 
    :class:`AtomArray`. The algorithm is able to identify canonical and
    non-canonical base pairs between the 5 common bases Adenine,
    Guanine, Thymine, Cytosine, and Uracil bound to Deoxyribose and
    Ribose.

    A standard reference frame for these bases as described in [2]_  has
    been implemented.

    The DSSR Criteria are as follows:

    (i) Distance between base origins <=15 Å

    (ii) Vertical separation between the base planes <=2.5 Å
    
    (iii) Angle between the base normal vectors <=65°

    (iv) Absence of stacking between the two bases

    (v) Presence of at least one hydrogen bond involving a base atom

    Parameters
    ----------
    atom_array : AtomArray
        The :class:`AtomArray` to find basepairs in.
    min_atoms_per_base : integer, optional (default: 3)
        The number of atoms a nucleotides' base must have to be 
        considered a candidate for a basepair.
    unique : bool, optional (default: True)
        If ``True``, each base is assumed to be only paired with one
        other base. If multiple pairings are plausible, the one with
        the shortest hydrogen bond is selected.
        
    Returns
    -------
    basepairs : ndarray, dtype=int, shape=(n,2)
        Each row is equivalent to one basepair and contains the first
        indices of the residues corresponding to each base.

    Notes
    -----
    If a base is incomplete but contains the minimum number of base-
    atoms specified, a superimposed standard base is used to emulate it.

    The vertical separation has been implemented as the rise parameter
    between the base triads [3]_.

    The presence of base stacking is assumed if the following criteria
    are met [4]_:

    (i) Distance between aromatic ring centers <=4.5 Å

    (ii) Angle between the ring normal vectors <=23°
    
    (iii) Angle between normalized distance vector between two ring
          centers and one normal vector <=40°

    Please note that ring normal vectors are assumed to be equal to the
    base normal vectors.

    For structures without hydrogens only the plausibility of hydrogen 
    bonds can be checked. A hydrogen bond is considered as plausible if
    a cutoff of 4.0 Å between a heteroatom that is bound to a hydrogen,
    that can act as hydrogen bond donor, and a heteroatom that can
    accept hydrogen bonds, is met [1]_.

    Examples
    --------
    Compute the basepairs for the structure with the PDB id 1QXB:
    
    >>> basepairs = base_pairs(atom_array)
    >>> print(basepairs)
    [[  0 723]
     [ 28 693]
     [ 61 660]
     [ 91 630]
     [124 598]
     [156 566]
     [188 534]
     [220 502]
     [252 469]
     [282 439]
     [315 406]
     [345 378]]

    References
    ----------
    
    .. [1] XJ Lu, HJ Bussemaker and WK Olson,
       "DSSR: an integrated software tool for dissecting the spatial
       structure of RNA."
       Nucleic acids research, 43(21), e142 (2015).

    .. [2] WK Olson, M Bansal and SK Burley et al.,
       "A standard reference frame for the description of nucleic acid
       base-pair geometry."
       J Mol Biol, 313(1), 229-237 (2001).

    .. [3] XJ Lu, MA El Hassan and CA Hunter,
        "Structure and conformation of helical nucleic acids: analysis
        program (SCHNAaP)."
        J Mol Biol, 273, 668-680 (1997).

    .. [4] HA Gabb, SR Sanghani and CH Robert et al.,
       "Finding and visualizing nucleic acid base stacking"
       J Mol Biol Graph, 14(1), 6-11 (1996).
    """

    # Get the basepair candidates according to a N/O cutoff distance,
    # where each base is identified as the first index of its respective
    # residue
    basepair_candidates = _get_proximate_basepair_candidates(atom_array)

    # Contains the plausible basepairs
    basepairs = []
    # Contains the distance of the shortest Hydrogen Bond for each
    # plausible basepair
    basepairs_hbonds = []

    for base1_index, base2_index in basepair_candidates:
        base1 = atom_array[_filter_residues(atom_array, base1_index)]
        base2 = atom_array[_filter_residues(atom_array, base2_index)]
        hbonds =  _check_dssr_criteria(
            (base1, base2), min_atoms_per_base, unique
        )
        if not hbonds == -1:
            basepairs.append((base1_index, base2_index))
            if unique:
                basepairs_hbonds.append(hbonds)

    basepair_array = np.array(basepairs)
    
    if unique:
        # Contains all non-unique basepairs that are flagged to be
        # removed
        to_remove = []

        # Get all bases that have non-unique pairing interactions
        base_indices, occurrences = np.unique(basepairs, return_counts=True)
        for base_index, occurrence in zip(base_indices, occurrences):
            if(occurrence > 1):
                # Write the non-unique basepairs to a dictionary as 
                # 'index: shortest_hbond_length'
                remove_candidates = {}
                for i, row in enumerate(
                    np.asarray(basepair_array == base_index)
                ):
                    if(np.any(row)):
                        remove_candidates[i] = basepairs_hbonds[i]
                # Flag all non-unique basepairs for removal except the
                # one that has the shortest hydrogen bond
                del remove_candidates[
                    min(remove_candidates, key=remove_candidates.get)
                ]
                to_remove += list(remove_candidates.keys())
        # Remove all flagged basepairs from the output `ndarray`
        basepair_array = np.delete(basepair_array, to_remove, axis=0)
    
    return basepair_array


def _check_dssr_criteria(basepair, min_atoms_per_base, unique):
    """
    Check the DSSR criteria of a potential basepair.
    
    Parameters
    ----------
    basepair : tuple (AtomArray, AtomArray)
        The two bases to check the criteria for as :class:`AtomArray`.
    min_atoms_per_base : integer
        The number of atoms a nucleotides' base must have to be 
        considered a candidate for a basepair.
    unique : bool
        If ``True``, the shortest hydrogen bond length between the bases
        is calculated for plausible basepairs.
        
    Returns
    -------
    satisfied : integer
        `>0` if the basepair satisfies the criteria and `-1` if it does
        not. If unique is ``True``, the shortest hydrogen bond length
        is returned for plausible basepairs.
    """

    # Contains the bases to be used for analysis. If the bases are 
    # incomplete, transformed standard bases are used. If they are 
    # complete, the original structure is used.
    transformed_bases = [None] * 2
    # Contains the hydrogen bond donor and acceptor heteroatoms as 
    # 'ndarray` with dtype=bool, boolean mask for heteroatoms which are
    # bound to a hydrogen that can act as a donor, boolean mask for
    # heteroatoms that can act as a hydrogen bond acceptor
    hbond_masks = [None] * 2
    # A list containing ndarray for each base with transformed
    # vectors from the standard base reference frame to the structures'
    # coordinates. The layout is as follows:
    #
    # [Origin coordinates]
    # [Base normal vector]
    # [SCHNAaP origin coordinates]
    # [Aromatic Ring Center coordinates]
    transformed_std_vectors = [None] * 2
    
    # Generate the data necessary for analysis of each base.
    for i in range(2):
        base_tuple = _match_base(basepair[i], min_atoms_per_base)
        
        if(base_tuple is None):
            return -1
        
        transformed_bases[i], hbond_masks[i], transformed_std_vectors[i] \
            = base_tuple
    
    # Criterion 1: Distance between orgins <=15 Å
    if not (distance(transformed_std_vectors[0][0,:],
            transformed_std_vectors[1][0,:]) <= 15):
        return -1
    # Criterion 2: Vertical separation <=2.5 Å
    #
    # Align the bases` normal vectors with the reference frame described
    # by the SCHNAaP algorithm
    normal_vector_schnaap = [None]*2
    for i in range(2):
        if(
            transformed_bases[i].res_name[0] not in (
                _adenine_containing_nucleotides \
                + _guanine_containing_nucleotides
            )
        ):
            normal_vector_schnaap[i] = (-1)*transformed_std_vectors[i][1,:]
        else:
            normal_vector_schnaap[i] = transformed_std_vectors[i][1,:]

    # Calculate the angle between normal vectors of the bases
    normal_vector_angle = np.arccos(np.dot(normal_vector_schnaap[0],
                                           normal_vector_schnaap[1]))
    # Calculate the orthonormal vector to the normal vectors of the
    # bases
    rotation_axis = np.cross(normal_vector_schnaap[0],
                                 normal_vector_schnaap[1])
    norm_vector(rotation_axis)
    # Rotate the base normal vectors by ± half the angle between the two
    # vectors
    rotated_normal_vector_0 = np.dot(
        _get_rotation_matrix(rotation_axis, normal_vector_angle/2),
        normal_vector_schnaap[0]
    )
    rotated_normal_vector_1 = np.dot(
        _get_rotation_matrix(rotation_axis, ((-1)*normal_vector_angle)/2),
        normal_vector_schnaap[1]
    )
    # Average and normalize the rotated vectors
    z_rot_average = (rotated_normal_vector_0 + rotated_normal_vector_1)/2
    norm_vector(z_rot_average)
    # Calculate the vector between the two origins    
    origin_vector = transformed_std_vectors[1][2,:] \
                    - transformed_std_vectors[0][2,:]
    
    # The dot product between the averaged rotated normal vectors and 
    # the vector between the two origins is the vertical separation
    if not abs(np.dot(origin_vector, z_rot_average)) <= 2.5:
        return -1
    
    # Criterion 3: Angle between normal vectors <=65°
    if not (np.arccos(np.dot(transformed_std_vectors[0][1,:],
                              transformed_std_vectors[1][1,:])) 
            >= ((115*np.pi)/180)):
        return -1
    
    # Criterion 4: Absence of stacking
    if _check_base_stacking(transformed_std_vectors):
        return -1
    
    # Criterion 5: Presence of at least one hydrogen bond
    #
    # Check if both bases came with hydrogens.
    if (("H" in transformed_bases[0].element)
        and ("H" in transformed_bases[1].element)):
        # For Structures that contain hydrogens, check for their 
        # presence directly.
        #
        # Default return value if no basepair is found
        hbonds = -1
        # Generate input atom array for ``hbond```
        potential_basepair = transformed_bases[0] + transformed_bases[1]

        # Iterate through output of ``hbond```
        for bond in hbond(
            potential_basepair,
            np.ones_like(potential_basepair, dtype=bool), 
            np.ones_like(potential_basepair, dtype=bool)
        ):
            if not unique:
                # If there is output but the uniqueness is not checked
                # return `1`
                return 1

            if (distance(potential_basepair[bond[0]].coord, 
                potential_basepair[bond[2]].coord) < hbonds) \
                or (hbonds == -1):
                # If the distance is smaller than previously found use
                # as output value
                hbonds = distance(potential_basepair[bond[0]].coord, 
                                  potential_basepair[bond[2]].coord)
        return hbonds

    else:
        # If the structure does not contain hydrogens, check for the
        # plausibility of hydrogen bonds between heteroatoms       
        return _check_hbonds(transformed_bases, hbond_masks, unique)  

def _check_hbonds(bases, hbond_masks, unique):
    """
    Check if hydrogen bonds are plausible between two bases. A cutoff
    of 4.0 Å between a heteroatom that is bound to a hydrogen, that can
    act as hydrogen bond donor, and a heteroatom that can accept
    hydrogen bonds, is used.
    
    Parameters
    ----------
    bases : list [AtomArray, AtomArray]
        The two bases to check for hydrogen bonds as :class:`AtomArray`.
    hbond_masks : list
        Contains the hydrogen bond donor and acceptor heteroatoms as 
        :class:`ndarray` with dtype=bool, boolean mask for heteroatoms
        which are bound to a hydrogen that can act as a donor, boolean
        mask for heteroatoms that can act as a hydrogen bond acceptor
    unique : bool
        If ``True``, the shortest hydrogen bond length between the bases
        is calculated for plausible basepairs.
        
    Returns
    -------
    plausible : integer
        `>0` if the basepair has plausible hydrogen bonds and `-1` if it
        does not. If unique is ``True``, the shortest hydrogen bond
        length is returned for plausible basepairs.
    """

    # Contains the length of plausible hydrogen bonds
    hbonds = []
    for donor_base, hbond_donor_mask, acceptor_base, hbond_acceptor_mask in \
        zip(bases, hbond_masks, reversed(bases), reversed(hbond_masks)):
        for donor_atom in donor_base[hbond_donor_mask[0]]:
            for acceptor_atom in acceptor_base[hbond_acceptor_mask[1]]:
                if(distance(acceptor_atom.coord, donor_atom.coord) <= 4.0):
                    if not unique:
                        # If a plausible hydrogen bond is found but the 
                        # uniqueness is not checked return `1`
                        return 1
                    hbonds.append(
                        distance(acceptor_atom.coord, donor_atom.coord)
                    )

    if len(hbonds) > 0:
        # Return the shortest hydrogen bond length
        if(bases[0].res_id[0] == 6 or bases[1].res_id[0] == 6):
            print(min(hbonds))
            print(str(bases[0].res_id[0]) + "und" + str(bases[1].res_id[0]))
        return min(hbonds)
        
    return -1


def _check_base_stacking(transformed_vectors):
    """
    Check for base stacking between two bases. 
    
    Parameters
    ----------
    transformed_vectors : list
        A list with transformed vectors as :class:`ndarray` for both 
        bases, origin coordinates, base normal vector, SCHNAaP origin,
        aromatic ring center coordinates
        
    Returns
    -------
    base_stacking : bool
        ``True`` if base stacking is detected and ``False`` if not
    """

    # Contains the normalized distance vectors between ring centers less
    # than 4.5 Å apart.
    normalized_distance_vectors = []

    # Criterion 1: Distance between aromatic ring centers <=4.5 Å
    wrongdistance = True
    for ring_center1 in transformed_vectors[0][3:][:]:
        for ring_center2 in transformed_vectors[1][3:][:]:
            if (distance(ring_center1, ring_center2) <= 4.5):
                wrongdistance = False
                normalized_distance_vectors.append(ring_center2 - ring_center1)
                norm_vector(normalized_distance_vectors[-1]) 
    if(wrongdistance == True):
        return False
    
    # Criterion 2: Angle between normal vectors or its supplement <=23°
    if (
            (np.arccos(np.dot(transformed_vectors[0][1,:],
                              transformed_vectors[1][1,:]))
            ) >= ((23*np.pi)/180)
            and (np.arccos(np.dot(transformed_vectors[0][1,:],
                              transformed_vectors[1][1,:]))
            ) <= ((157*np.pi)/180)
    ):
        return False
    
    # Criterion 3: Angle between one normalized distance vector and one 
    # normal vector or its supplement <=40°
    for vector in transformed_vectors:
        for normalized_dist_vector in normalized_distance_vectors:    
            if (
                (np.arccos(np.dot(vector[1,:], normalized_dist_vector))
                <= ((40*np.pi)/180))
                or (np.arccos(np.dot(vector[1,:], normalized_dist_vector))
                >= ((120*np.pi)/180))
            ):
                return True
    
    return False


def _match_base(nucleotide, min_atoms_per_base):
    """
    Match the nucleotide to a corresponding standard base.
    
    Parameters
    ----------
    nucleotide : AtomArray
        The nucleotide to be matched to a standard base.
    min_atoms_per_base : integer
        The number of atoms a base must have to be considered a 
        candidate for a basepair.
        
    Returns
    -------
    return_base or None : AtomArray
        The base of the nucleotide. If the given base is incomplete but
        contains the minimum number of atoms specified a superimposed
        standard base is returned. Else ``None`` is returned.
    return_hbond_masks : list
        The hydrogen bond donor and acceptor heteroatoms as 
        :class:`ndarray` with `dtype=bool`, boolean mask for heteroatoms
        which are bound to a hydrogen that can act as a donor, boolean
        mask for heteroatoms that can act as a hydrogen bond acceptor
    vectors : ndarray, dtype=float, shape=(n,3)
        Transformed standard vectors, origin coordinates, base normal
        vector, aromatic ring center coordinates
    """
    return_hbond_masks = [None] * 2
    # Standard vectors containing the origin and the base normal vectors
    vectors = np.array([[0, 0, 0], [0, 0, 1]], np.float)

    # Check base type and match standard base.
    if(nucleotide.res_name[0] in _adenine_containing_nucleotides):
        std_base = _std_adenine
        std_ring_centers = _std_adenine_ring_centers
        std_hbond_masks = _std_adenine_hbond_masks
    elif(nucleotide.res_name[0] in _thymine_containing_nucleotides):
        std_base = _std_thymine
        std_ring_centers = _std_thymine_ring_centers
        std_hbond_masks = _std_thymine_hbond_masks
    elif(nucleotide.res_name[0] in _cytosine_containing_nucleotides):
        std_base = _std_cytosine
        std_ring_centers = _std_cytosine_ring_centers
        std_hbond_masks = _std_cytosine_hbond_masks
    elif(nucleotide.res_name[0] in _guanine_containing_nucleotides):
        std_base = _std_guanine
        std_ring_centers = _std_guanine_ring_centers
        std_hbond_masks = _std_guanine_hbond_masks
    elif(nucleotide.res_name[0] in _uracil_containing_nucleotides):
        std_base = _std_uracil
        std_ring_centers = _std_uracil_ring_centers
        std_hbond_masks = _std_uracil_hbond_masks
    else:
        warnings.warn(
            f"Base Type {nucleotide.res_name[0]} not supported. Unable to "
            "check for basepair", UnexpectedStructureWarning
        )
        return None

    # Check if the structure uses PDBv3 or PDBv2 atom nomenclature.
    if (np.sum(np.in1d(std_base[1].atom_name, nucleotide.atom_name))
        > np.sum(np.in1d(std_base[0].atom_name, nucleotide.atom_name))):
        std_base = std_base[1]
    else:
        std_base = std_base[0]

    # Add the ring centers to the array of vectors to be transformed.
    vectors = np.vstack((vectors, std_ring_centers))
    
    # Match the selected std_base to the base.
    fitted, transformation = superimpose(
        nucleotide[np.in1d(nucleotide.atom_name, std_base.atom_name)],
        std_base[np.in1d(std_base.atom_name, nucleotide.atom_name)]
    )

    # Transform the vectors
    trans1, rot, trans2 = transformation
    vectors += trans1
    vectors  = np.dot(rot, vectors.T).T
    vectors += trans2   
    # Normalize the base-normal-vector   
    vectors[1,:] = vectors[1,:]-vectors[0,:]
    norm_vector(vectors[1,:])

    # Investigate the completeness of the base:
    # 
    # A difference in length of zero means the base contains all atoms
    # of the std_base          
    length_difference = len(std_base) - len(fitted)
    
    if(length_difference > 0 and len(fitted) >= min_atoms_per_base):
        # If the base is incomplete but contains 3 or more atoms of the 
        # std_base, transform the complete std_base and use it to
        # approximate the base.
        warnings.warn(
            f"Base with res_id {nucleotide.res_id[0]} and chain_id " 
            f"{nucleotide.chain_id[0]} is not complete. Attempting to "
            "emulate with std_base.", IncompleteStructureWarning
        )
        return_base = superimpose_apply(std_base, transformation)
        return_hbond_masks = std_hbond_masks
    elif (length_difference > 0):
        # If the base is incomplete and contains less than 3 atoms of 
        # the std_base, throw warning
        warnings.warn(
            f"Base with res_id {nucleotide.res_id[0]} and chain_id "
            f"{nucleotide.chain_id[0]} has an overlap with std_base "
            "which is less than 3 atoms. Unable to check for basepair."
            , IncompleteStructureWarning
        )
        return None
    else:
        # If the base is complete use the base for further calculations.
        #
        # Generate a boolean mask containing only the base atoms and
        # their hydrogens (if available), disregarding the sugar atoms
        # and the phosphate backbone.
        base_atom_mask = np.ones(len(nucleotide), dtype=bool)
        for i in range(len(nucleotide)):
            if (
                ("'" in nucleotide[i].atom_name)
                or ("*" in nucleotide[i].atom_name)
                or ((nucleotide[i].atom_name not in std_base.atom_name)
                    and (nucleotide[i].element != "H"))
            ):
                base_atom_mask[i] = False
        
        # Create boolean masks for the AtomArray containing the bases` 
        # heteroatoms (or the usually attached hydrogens), which can act 
        # as Hydrogen Bond Donors or Acceptors respectively, using the
        # std_base as a template.
        for i in range(2):
            return_hbond_masks[i] = _filter_atom_type(
                nucleotide[base_atom_mask], 
                std_base[std_hbond_masks[i]].atom_name
            )
        return_base = nucleotide[base_atom_mask]

    return return_base, return_hbond_masks, vectors


def _get_proximate_basepair_candidates(atom_array, cutoff = 4):
    """
    Filter for potential basepairs based on the distance between the
    nitrogen and oxygen atoms.
    
    Parameters
    ----------
    atom_array : AtomArray
        The :class:`AtomArray`` to find basepair candidates in.
    cutoff : integer
        The maximum distance of the N and O Atoms for two bases
        to be considered a basepair candidate.
        
    Returns
    -------
    basepair_candidates : list [(integer, integer), ...]
        Contains the basepair candidates, ``tuple`` of the first indices 
        of the corresponding residues.
    """

    # Get a boolean mask for the N and O atoms
    NOmask = (filter_nucleotides(atom_array) 
              & np.isin(atom_array.element, ["N", "O"]))
    # Get the indices of the N and O atoms that are within the maximum
    # cutoff of each other
    indices = CellList( 
        atom_array, cutoff, selection=NOmask
    ).get_atoms(atom_array.coord[NOmask], cutoff)
    
    # Loop through the indices of potential partners
    basepair_candidates = []
    for candidate, partners in zip(np.argwhere(NOmask), indices):
        for partner in partners:
            if partner == -1:
                break
            # Find the indices of the first atom of the residues
            candidate_res_start = np.where(
                (atom_array.res_id == atom_array[candidate].res_id)
                & (atom_array.chain_id == atom_array[candidate].chain_id)
            )[0][0]
            partner_res_start = np.where(
                (atom_array.res_id == atom_array[partner].res_id)
                & (atom_array.chain_id == atom_array[partner].chain_id)
            )[0][0]
            # If the basepair candidate is not already in the output
            # list, append to the output list
            if (
                ((partner_res_start, candidate_res_start) \
                not in basepair_candidates)
                and ((candidate_res_start, partner_res_start) \
                not in basepair_candidates)
                and not (candidate_res_start == partner_res_start)
            ):
                basepair_candidates.append(
                    (candidate_res_start, partner_res_start)
                )
    return basepair_candidates


def _filter_atom_type(atom_array, atom_names):
    """
    Get all atoms with specified atom names.
    
    Parameters
    ----------
    atom_array : AtomArray
        The :class:`AtomArray` to filter.
    atom_names : array_like
        The desired atom names.
        
    Returns
    -------
    filter : ndarray, dtype=bool
        This array is ``True`` for all indices in the :class:`AtomArray`
        , where the atom has the desired atom names.
    """
    return (np.isin(atom_array.atom_name, atom_names)
            & (atom_array.res_id != -1))


def _filter_residues(atom_array, index):
    """
    Get all atoms of a residue that an atom at the specified index 
    belongs to.
    
    Parameters
    ----------
    atom_array : AtomArray
        The :class:`AtomArray` to filter.
    index : integer
        The index of an atom of the desired residue.
        
    Returns
    -------
    filter : ndarray, dtype=bool
        This array is ``True`` for all indices in the :class:`AtomArray`
        where the atom has the desired ``residue_name`` and 
        ``chain_id``.
    """
    return (np.isin(atom_array.res_id, atom_array[int(index)].res_id)
            & np.isin(atom_array.chain_id, atom_array[int(index)].chain_id))
        
def _get_rotation_matrix(axis, angle):
    """
    Get a rotation matrix for a rotation in 3d space.
    
    Parameters
    ----------
    axis : ndarray, dtype=float, shape=(3,)
        The axis the rotation is performed on.
    angle : float
        The angle of the rotation in radians.
        
    Returns
    -------
    rotation_matrix : ndarray, dtype=float, shape=(3,3)
        The rotation matrix for the desired rotation.
    """
    rotation_matrix = np.zeros((3,3), dtype=float)
    n1, n2, n3 = axis
    normal_cos = np.cos(angle)
    inverse_cos = 1 - normal_cos

    rotation_matrix[0,0] = ((n1*n1)*inverse_cos) + normal_cos
    rotation_matrix[0,1] = ((n1*n2)*inverse_cos) - (n3*np.sin(angle))
    rotation_matrix[0,2] = ((n1*n3)*inverse_cos) + (n2*np.sin(angle))
    rotation_matrix[1,0] = ((n2*n1)*inverse_cos) + (n3*np.sin(angle))
    rotation_matrix[1,1] = ((n2*n2)*inverse_cos) + normal_cos
    rotation_matrix[1,2] = ((n2*n3)*inverse_cos) - (n1*np.sin(angle))
    rotation_matrix[2,0] = ((n3*n1)*inverse_cos) - (n2*np.sin(angle))
    rotation_matrix[2,1] = ((n3*n2)*inverse_cos) + (n1*np.sin(angle))
    rotation_matrix[2,2] = ((n3*n3)*inverse_cos) + normal_cos

    return rotation_matrix