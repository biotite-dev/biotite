# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for basepair identification.
"""

__name__ = "biotite.structure"
__author__ = "Tom David Müller"
__all__ = ["base_pairs", "map_nucleotide"]

import numpy as np
import warnings
from .atoms import Atom, array
from .superimpose import superimpose, superimpose_apply
from .filter import filter_nucleotides
from .celllist import CellList
from .hbond import hbond
from .error import IncompleteStructureWarning, UnexpectedStructureWarning
from .util import distance, norm_vector
from .residues import get_residue_starts_for, get_residue_masks
from .info.standardize import standardize_order
from .compare import rmsd


def _get_std_adenine():
    """
    Get standard base variables for adenine.

    Returns
    -------
    standard_base : AtomArray
        Standard coordinates nomenclature of the adenine base as
        :class:`AtomArray` with nomenclature of PDB File Format V3
    coordinates : tuple (ndarray, ndarray, ndarray, dtype=float)
        :class:`ndarray` containing the center according to the SCHNaP-
        paper referenced in the function ``base_pairs``,
        :class:`ndarray` containing the coordinates of the pyrimidine
        ring center, :class:`ndarray` containing the coordinates of the
        imidazole ring center
    """
    atom1 =  Atom([-1.291, 4.498, 0.000], atom_name="N9",  res_name="A")
    atom2 =  Atom([0.024, 4.897, 0.000],  atom_name="C8",  res_name="A")
    atom3 =  Atom([0.877, 3.902, 0.000],  atom_name="N7",  res_name="A")
    atom4 =  Atom([0.071, 2.771, 0.000],  atom_name="C5",  res_name="A")
    atom5 =  Atom([0.369, 1.398, 0.000],  atom_name="C6",  res_name="A")
    atom6 =  Atom([1.611, 0.909, 0.000],  atom_name="N6",  res_name="A")
    atom7 =  Atom([-0.668, 0.532, 0.000], atom_name="N1",  res_name="A")
    atom8 =  Atom([-1.912, 1.023, 0.000], atom_name="C2",  res_name="A")
    atom9 = Atom([-2.320, 2.290, 0.000], atom_name="N3",  res_name="A")
    atom10 = Atom([-1.267, 3.124, 0.000], atom_name="C4",  res_name="A")
    adenine = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8,
         atom9, atom10]
    )

    # Get the midpoint between the N1 and C4 atoms
    midpoint = np.mean([atom7.coord, atom10.coord], axis=-2)
    # Calculate the coordinates of the aromatic ring centers
    pyrimidine_center = np.mean(
        [atom4.coord, atom5.coord, atom7.coord,
         atom8.coord, atom9.coord, atom10.coord], axis=-2
    )
    imidazole_center = np.mean(
        [atom1.coord, atom2.coord, atom3.coord,
         atom4.coord, atom10.coord], axis=-2
    )

    return adenine, (midpoint, pyrimidine_center, imidazole_center)


def _get_std_cytosine():
    """
    Get standard base variables for cytosine.

    Returns
    -------
   standard_base : AtomArray
        Standard coordinates nomenclature of the cytosine base as
        :class:`AtomArray` with nomenclature of PDB File Format V3
    coordinates : tuple (ndarray, ndarray, dtype=float)
        :class:`ndarray` containing the center according to the SCHNaP-
        paper referenced in the function ``base_pairs``,
        :class:`ndarray` containing the coordinates of the pyrimidine
        ring center
    """
    atom1 = Atom([-1.285, 4.542, 0.000], atom_name="N1",  res_name="C")
    atom2 = Atom([-1.472, 3.158, 0.000], atom_name="C2",  res_name="C")
    atom3 = Atom([-2.628, 2.709, 0.000], atom_name="O2",  res_name="C")
    atom4 = Atom([-0.391, 2.344, 0.000], atom_name="N3",  res_name="C")
    atom5 = Atom([0.837, 2.868, 0.000],  atom_name="C4",  res_name="C")
    atom6 = Atom([1.875, 2.027, 0.000],  atom_name="N4",  res_name="C")
    atom7 = Atom([1.056, 4.275, 0.000],  atom_name="C5",  res_name="C")
    atom8 = Atom([-0.023, 5.068, 0.000], atom_name="C6",  res_name="C")
    cytosine = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8]
    )

    # Get the midpoint between the N3 and C6 atoms
    midpoint = np.mean([atom4.coord, atom8.coord], axis=-2)
    # Calculate the coordinates of the aromatic ring center
    pyrimidine_center = np.mean(
        [atom1.coord, atom2.coord, atom4.coord,
         atom5.coord, atom7.coord, atom8.coord], axis=-2
    )

    return cytosine, (midpoint, pyrimidine_center)


def _get_std_guanine():
    """
    Get standard base variables for guanine.

    Returns
    -------
    standard_base : AtomArray
        Standard coordinates nomenclature of the guanine base as
        :class:`AtomArray` with nomenclature of PDB File Format V3
    coordinates : tuple (ndarray, ndarray, ndarray, dtype=float)
        :class:`ndarray` containing the center according to the SCHNaP-
        paper referenced in the function ''base_pairs'',
        :class:`ndarray` containing the coordinates of the pyrimidine
        ring center, :class:`ndarray` containing the coordinates of the
        imidazole ring center
    """
    atom1 =  Atom([-1.289, 4.551, 0.000],  atom_name="N9",  res_name="G")
    atom2 =  Atom([0.023, 4.962, 0.000],   atom_name="C8",  res_name="G")
    atom3 =  Atom([0.870, 3.969, 0.000],   atom_name="N7",  res_name="G")
    atom4 =  Atom([0.071, 2.833, 0.000],   atom_name="C5",  res_name="G")
    atom5 =  Atom([0.424, 1.460, 0.000],   atom_name="C6",  res_name="G")
    atom6 =  Atom([1.554, 0.955, 0.000],   atom_name="O6",  res_name="G")
    atom7 =  Atom([-0.700, 0.641, 0.000],  atom_name="N1",  res_name="G")
    atom8 =  Atom([-1.999, 1.087, 0.000],  atom_name="C2",  res_name="G")
    atom9 = Atom([-2.949, 0.139, -0.001], atom_name="N2",  res_name="G")
    atom10 = Atom([-2.342, 2.364, 0.001],  atom_name="N3",  res_name="G")
    atom11 = Atom([-1.265, 3.177, 0.000],  atom_name="C4",  res_name="G")
    guanine = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8,
         atom9, atom10, atom11]
    )

    # Get the midpoint between the N1 and C4 atoms
    midpoint = np.mean([atom7.coord, atom11.coord], axis=-2)
    # Calculate the coordinates of the aromatic ring centers
    pyrimidine_center = np.mean(
        [atom4.coord, atom5.coord, atom7.coord,
         atom8.coord, atom10.coord, atom11.coord], axis=-2
    )
    imidazole_center = np.mean(
        [atom1.coord, atom2.coord, atom3.coord,
         atom4.coord, atom11.coord], axis=-2
    )

    return guanine, (midpoint, pyrimidine_center, imidazole_center)


def _get_std_thymine():
    """
    Get standard base variables for thymine.

    Returns
    -------
    standard_base : AtomArray
        Standard coordinates nomenclature of the thymine base as
        :class:`AtomArray` with nomenclature of PDB File Format V3
    coordinates : tuple (ndarray, ndarray, dtype=float)
        :class:`ndarray` containing the center according to the SCHNaP-
        paper referenced in the function ``base_pairs``,
        :class:`ndarray` containing the coordinates of the pyrimidine
        ring center
    """
    atom1 =  Atom([-1.284, 4.500, 0.000], atom_name="N1",  res_name="T")
    atom2 =  Atom([-1.462, 3.135, 0.000], atom_name="C2",  res_name="T")
    atom3 =  Atom([-2.562, 2.608, 0.000], atom_name="O2",  res_name="T")
    atom4 =  Atom([-0.298, 2.407, 0.000], atom_name="N3",  res_name="T")
    atom5 =  Atom([0.994, 2.897, 0.000],  atom_name="C4",  res_name="T")
    atom6 =  Atom([1.944, 2.119, 0.000],  atom_name="O4",  res_name="T")
    atom7 =  Atom([1.106, 4.338, 0.000],  atom_name="C5",  res_name="T")
    atom8 =  Atom([2.466, 4.961, 0.001],  atom_name="C7", res_name="T")
    atom9 = Atom([-0.024, 5.057, 0.000], atom_name="C6",  res_name="T")
    thymine = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9]
    )

    # Get the midpoint between the N3 and C6 atoms
    midpoint = np.mean([atom4.coord, atom9.coord], axis=-2)
    # Calculate the coordinates of the aromatic ring center
    pyrimidine_center = np.mean(
        [atom1.coord, atom2.coord, atom4.coord,
         atom5.coord, atom7.coord, atom9.coord], axis=-2
    )

    return thymine, (midpoint, pyrimidine_center)


def _get_std_uracil():
    """
    Get standard base variables for uracil.

    Returns
    -------
    standard_base : AtomArray
        Standard coordinates nomenclature of the uracil base as
        :class:`AtomArray` with nomenclature of PDB File Format V3
    coordinates : tuple (ndarray, ndarray, dtype=float)
        :class:`ndarray` containing the center according to the SCHNaP-
        paper referenced in the function ``base_pairs``,
        :class:`ndarray` containing the coordinates of the pyrimidine
        ring center
    """
    atom1 = Atom([-1.284, 4.500, 0.000], atom_name="N1",  res_name="U")
    atom2 = Atom([-1.462, 3.131, 0.000], atom_name="C2",  res_name="U")
    atom3 = Atom([-2.563, 2.608, 0.000], atom_name="O2",  res_name="U")
    atom4 = Atom([-0.302, 2.397, 0.000], atom_name="N3",  res_name="U")
    atom5 = Atom([0.989, 2.884, 0.000],  atom_name="C4",  res_name="U")
    atom6 = Atom([1.935, 2.094, -0.001], atom_name="O4",  res_name="U")
    atom7 = Atom([1.089, 4.311, 0.000],  atom_name="C5",  res_name="U")
    atom8 = Atom([-0.024, 5.053, 0.000], atom_name="C6",  res_name="U")
    uracil = array(
        [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8]
    )

    # Get the midpoint between the N3 and C6 atoms
    midpoint = np.mean([atom4.coord, atom8.coord], axis=-2)
    # Calculate the coordinates of the aromatic ring center
    pyrimidine_center = np.mean(
        [atom1.coord, atom2.coord, atom4.coord,
         atom5.coord, atom7.coord, atom8.coord], axis=-2
    )

    return uracil, (midpoint, pyrimidine_center)


_std_adenine, _std_adenine_ring_centers  = _get_std_adenine()
_std_cytosine, _std_cytosine_ring_centers = _get_std_cytosine()
_std_guanine, _std_guanine_ring_centers = _get_std_guanine()
_std_thymine, _std_thymine_ring_centers = _get_std_thymine()
_std_uracil, _std_uracil_ring_centers = _get_std_uracil()

_adenine_containing_nucleotides = ["A", "DA"]
_thymine_containing_nucleotides = ["T", "DT"]
_cytosine_containing_nucleotides = ["C", "DC"]
_guanine_containing_nucleotides = ["G", "DG"]
_uracil_containing_nucleotides = ["U", "DU"]


def base_pairs(atom_array, min_atoms_per_base = 3, unique = True):
    """
    Use DSSR criteria to find the basepairs in an :class:`AtomArray`.

    The algorithm is able to identify canonical and non-canonical
    base pairs. between the 5 common bases Adenine, Guanine, Thymine,
    Cytosine, and Uracil bound to Deoxyribose and Ribose.
    Each Base is mapped to the 5 common bases Adenine, Guanine, Thymine,
    Cytosine, and Uracil in a standard reference frame described in
    [1]_ using :func:`map_nucleotide()`.

    The DSSR Criteria are as follows [2]_ :

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
        other base. If multiple pairings are plausible, the pairing with
        the most hydrogen bonds is selected.

    Returns
    -------
    basepairs : ndarray, dtype=int, shape=(n,2)
        Each row is equivalent to one basepair and contains the first
        indices of the residues corresponding to each base.

    Notes
    -----
    The vertical separation is implemented as the scalar
    projection of the distance vectors between the base origins
    according to [3]_ onto the averaged base normal vectors.

    The presence of base stacking is assumed if the following criteria
    are met [4]_:

    (i) Distance between aromatic ring centers <=4.5 Å

    (ii) Angle between the ring normal vectors <=23°

    (iii) Angle between normalized distance vector between two ring
          centers and one normal vector <=40°

    Please note that ring normal vectors are assumed to be equal to the
    base normal vectors.

    For structures without hydrogens the accuracy of the algorithm is
    limited as the hydrogen bonds can be only checked be checked for
    plausibility.
    A hydrogen bond is considered as plausible if a cutoff of 3.6 Å
    between N/O atom pairs is met. 3.6Å was chosen as hydrogen bonds are
    typically 1.5-2.5Å in length. N-H and O-H bonds have a length of
    1.00Å and 0.96Å respectively. Thus, including some buffer, a 3.6Å
    cutoff should cover all hydrogen bonds.

    Examples
    --------
    Compute the basepairs for the structure with the PDB id 1QXB:

    >>> from os.path import join
    >>> dna_helix = load_structure(join(path_to_structures, "1qxb.cif"))
    >>> basepairs = base_pairs(dna_helix)
    >>> print(dna_helix[basepairs].res_name)
    [['DC' 'DG']
     ['DG' 'DC']
     ['DC' 'DG']
     ['DG' 'DC']
     ['DA' 'DT']
     ['DA' 'DT']
     ['DT' 'DA']
     ['DT' 'DA']
     ['DC' 'DG']
     ['DG' 'DC']
     ['DC' 'DG']
     ['DG' 'DC']]

    References
    ----------

    .. [1] WK Olson, M Bansal and SK Burley et al.,
       "A standard reference frame for the description of nucleic acid
       base-pair geometry."
       J Mol Biol, 313(1), 229-237 (2001).

    .. [2] XJ Lu, HJ Bussemaker and WK Olson,
       "DSSR: an integrated software tool for dissecting the spatial
       structure of RNA."
       Nucleic Acids Res, 43(21), e142 (2015).

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
    # Contains the number of hydrogens for each plausible basepair
    basepairs_hbonds = []

    for (base1_index, base2_index), n_o_matches in basepair_candidates.items():
        base1_mask, base2_mask = get_residue_masks(
            atom_array, (base1_index, base2_index)
        )
        base1 = atom_array[base1_mask]
        base2 = atom_array[base2_mask]
        hbonds =  _check_dssr_criteria(
            (base1, base2), min_atoms_per_base, unique
        )
        # If no hydrogens are present use the number N/O pairs to
        # decide between multiple pairing posibilities.
        if hbonds is None:
            # Each N/O-pair is detected twice. Thus, the number of
            # matches must be divided by two.
            hbonds = n_o_matches/2
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
                # 'index: number of hydrogen bonds'
                remove_candidates = {}
                for i, row in enumerate(
                    np.asarray(basepair_array == base_index)
                ):
                    if(np.any(row)):
                        remove_candidates[i] = basepairs_hbonds[i]
                # Flag all non-unique basepairs for removal except the
                # one that has the most hydrogen bonds
                del remove_candidates[
                    max(remove_candidates, key=remove_candidates.get)
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
    min_atoms_per_base : int
        The number of atoms a nucleotides' base must have to be
        considered a candidate for a basepair.
    unique : bool
        If ``True``, the shortest hydrogen bond length between the bases
        is calculated for plausible basepairs.

    Returns
    -------
    satisfied : int
        `> 0` if the basepair satisfies the criteria and `-1`,
        if it does not.
        If unique is ``True``, the number of hydrogen bonds is
        returned for plausible basepairs.
    """

    # Contains the bases to be used for analysis. If the bases are
    # incomplete, transformed standard bases are used. If they are
    # complete, the original structure is used.
    transformed_bases = [None] * 2
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

        transformed_bases[i], transformed_std_vectors[i] = base_tuple

    origins = np.vstack((transformed_std_vectors[0][0],
                         transformed_std_vectors[1][0]))
    normal_vectors = np.vstack((transformed_std_vectors[0][1],
                                transformed_std_vectors[1][1]))
    schnaap_origins = np.vstack((transformed_std_vectors[0][2],
                                 transformed_std_vectors[1][2]))
    aromatic_ring_centers = [transformed_std_vectors[0][3:],
                                       transformed_std_vectors[1][3:]]

    # Criterion 1: Distance between orgins <=15 Å
    if not (distance(origins[0], origins[1]) <= 15):
        return -1

    # Criterion 2: Vertical separation <=2.5 Å
    #
    # Average the base normal vectors. If the angle between the vectors
    # is >=90°, flip one vector before averaging
    mean_normal_vector = (
        normal_vectors[0] + (normal_vectors[1] * np.sign(np.dot(
            normal_vectors[0], normal_vectors[1]
        )))
    ) / 2
    norm_vector(mean_normal_vector)
    # Calculate the distance vector between the two SCHNAaP origins
    origin_distance_vector = schnaap_origins[1] - schnaap_origins[0]

    # The scalar projection of the distance vector between the two
    # origins onto the averaged normal vectors is the vertical
    # seperation
    if not abs(np.dot(origin_distance_vector, mean_normal_vector)) <= 2.5:
        return -1

    # Criterion 3: Angle between normal vectors <=65°
    if not (np.arccos(np.dot(normal_vectors[0], normal_vectors[1]))
            >= ((115*np.pi)/180)):
        return -1

    # Criterion 4: Absence of stacking
    if _check_base_stacking(aromatic_ring_centers, normal_vectors):
        return -1

    # Criterion 5: Presence of at least one hydrogen bond
    #
    # Check if both bases came with hydrogens.
    if (("H" in transformed_bases[0].element)
        and ("H" in transformed_bases[1].element)):
        # For Structures that contain hydrogens, check for their
        # presence directly.
        #
        # Generate input atom array for ``hbond```
        potential_basepair = transformed_bases[0] + transformed_bases[1]

        # Get the number of hydrogen bonds
        bonds = len(hbond(
            potential_basepair,
            np.ones_like(potential_basepair, dtype=bool),
            np.ones_like(potential_basepair, dtype=bool)
        ))

        if bonds > 0:
            return bonds

        return -1

    else:
        # If the structure does not contain hydrogens return None
        return None


def _check_base_stacking(aromatic_ring_centers, normal_vectors):
    """
    Check for base stacking between two bases.

    Parameters
    ----------
    aromatic_ring_centers : list [ndarray, ndarray]
        A list with the aromatic ring center coordinates as
        :class:`ndarray`. Each row represents a ring center.
    normal_vectors : ndarray shape=(2, 3)
        The normal vectors of the bases.

    Returns
    -------
    base_stacking : bool
        ``True`` if base stacking is detected and ``False`` if not
    """

    # Contains the normalized distance vectors between ring centers less
    # than 4.5 Å apart.
    normalized_distance_vectors = []

    # Criterion 1: Distance between aromatic ring centers <=4.5 Å
    wrong_distance = True
    for ring_center1 in aromatic_ring_centers[0]:
        for ring_center2 in aromatic_ring_centers[1]:
            if (distance(ring_center1, ring_center2) <= 4.5):
                wrong_distance = False
                normalized_distance_vectors.append(ring_center2 - ring_center1)
                norm_vector(normalized_distance_vectors[-1])
    if wrong_distance:
        return False

    # Criterion 2: Angle between normal vectors or its supplement <=23°
    if (
            (np.arccos(np.dot(normal_vectors[0], normal_vectors[1]))
            >= ((23*np.pi)/180))
            and (np.arccos(np.dot(normal_vectors[0], normal_vectors[1]))
            <= ((157*np.pi)/180))
    ):
        return False

    # Criterion 3: Angle between one normalized distance vector and one
    # normal vector or its supplement <=40°
    for normal_vector in normal_vectors:
        for normalized_dist_vector in normalized_distance_vectors:
            if (
                (np.arccos(np.dot(normal_vector, normalized_dist_vector))
                <= ((40*np.pi)/180))
                or (np.arccos(np.dot(normal_vector, normalized_dist_vector))
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
        mask for heteroatoms that can act as a hydrogen bond acceptor.
    vectors : ndarray, dtype=float, shape=(n,3)
        Transformed standard vectors, origin coordinates, base normal
        vector, aromatic ring center coordinates.
    """

    # Standard vectors containing the origin and the base normal vectors
    vectors = np.array([[0, 0, 0], [0, 0, 1]], np.float)

    # Map the nucleotide to a reference base
    base_tuple = map_nucleotide(nucleotide, min_atoms_per_base)

    if base_tuple is None:
        return None

    one_letter_code, _ = base_tuple
    if (one_letter_code == 'A'):
        std_base = _std_adenine
        std_ring_centers = _std_adenine_ring_centers
    elif (one_letter_code == 'T'):
        std_base = _std_thymine
        std_ring_centers = _std_thymine_ring_centers
    elif (one_letter_code == 'C'):
        std_base = _std_cytosine
        std_ring_centers = _std_cytosine_ring_centers
    elif (one_letter_code == 'G'):
        std_base = _std_guanine
        std_ring_centers = _std_guanine_ring_centers
    elif (one_letter_code == 'U'):
        std_base = _std_uracil
        std_ring_centers = _std_uracil_ring_centers

    # Add the ring centers to the array of vectors to be transformed.
    vectors = np.vstack((vectors, std_ring_centers))

    # Select the matching atoms of the nucleotide and the standard base
    nucleotide_matched = nucleotide[
        np.isin(nucleotide.atom_name, std_base.atom_name)
    ]
    std_base_matched = std_base[
        np.isin(std_base.atom_name, nucleotide.atom_name)
    ]

    # Reorder the atoms of the nucleotide to obtain the standard RCSB
    # PDB atom order
    nucleotide_matched = nucleotide_matched[standardize_order(
        nucleotide_matched
    )]

    # Match the selected std_base to the base.
    _, transformation = superimpose(nucleotide_matched, std_base_matched)

    # Transform the vectors
    trans1, rot, trans2 = transformation
    vectors += trans1
    vectors  = np.dot(rot, vectors.T).T
    vectors += trans2
    # Normalize the base-normal-vector
    vectors[1,:] = vectors[1,:]-vectors[0,:]
    norm_vector(vectors[1,:])

    # Disregard the phosphate-backbone
    nucleotide = nucleotide[~ np.isin(
        nucleotide, ["O5'", "P", "OP1", "OP2", "OP3", "HOP2", "HOP3"]
    )]

    return nucleotide, vectors

def map_nucleotide(residue, min_atoms_per_base=3, rmsd_cutoff=0.28):
    # Check if the residue is a 'standard' nucleotide
    if residue.res_name[0] in (_thymine_containing_nucleotides +
        _guanine_containing_nucleotides + _uracil_containing_nucleotides
        + _cytosine_containing_nucleotides + _adenine_containing_nucleotides
    ):
        return residue.res_name[0][-1], True

    # List of the standard bases for easy iteration
    std_base_list = [
        _std_adenine, _std_thymine, _std_cytosine, _std_guanine,
        _std_uracil
    ]

    # The number of matched atoms for each 'standard' base
    matched_atom_no = []

    # Count the number of matching atoms with the reference bases
    for ref_base in std_base_list:
        matched_atom_no.append(np.sum(
            np.isin(ref_base.atom_name, residue.atom_name)
        ))

    if max(matched_atom_no) < min_atoms_per_base:
        warnings.warn(
            warnings.warn(
            f"Base with res_id {residue.res_id[0]} and chain_id "
            f"{residue.chain_id[0]} has an overlap with the reference "
            f"bases which is less than {min_atoms_per_base} atoms."
            f"Unable to map nucleotide.",
            IncompleteStructureWarning
        )
        )
        return None

    # The one letter code of the best matching reference base
    best_base = None

    # Iterate through the reference bases with the maximum number of
    # matching atoms
    for ref_base in np.array(std_base_list)[
        np.array(matched_atom_no) == max(matched_atom_no)
    ]:
        # Copy the residue as the res_name property of the ``AtomArray``
        # has to be modified for later function calls.
        nuc = residue.copy()

        # Select the matching atoms of the nucleotide and the reference
        # base
        nuc = nuc[
            np.isin(nuc.atom_name, ref_base.atom_name)
        ]
        ref_base_matched = ref_base[
            np.isin(ref_base.atom_name, nuc.atom_name)
        ]

        # Set the res_name property to the same as the reference base.
        # This is a requirement for ``standardize_order``
        nuc.res_name = ref_base_matched.res_name
        # Reorder the atoms of the nucleotide to obtain the standard
        # RCSB PDB atom order. If a residue contains multiple Atoms with
        # the same ``atom_name`` an exception is thrown by
        # ``standardize_order``. The Exception is caught and the
        # selected reference is disregarded
        try:
            nuc = nuc[standardize_order(nuc)]
        except Exception:
            continue

        # Superimpose the nucleotide to the reference base
        fitted, _ = superimpose(ref_base_matched, nuc)

        # If the RMSD is lower than the specified cutoff or better than
        # a previous found reference, the current reference is selected
        # as best base
        if(rmsd(fitted, ref_base_matched) < rmsd_cutoff):
            rmsd_cutoff = rmsd(fitted, ref_base_matched)
            best_base = ref_base_matched.res_name[0][-1]

    if best_base is None:
        warnings.warn(
            f"Base Type {residue.res_name[0]} not supported. "
            f"Unable to check for basepair",
            UnexpectedStructureWarning
        )
        return None

    return best_base, False


def _get_proximate_basepair_candidates(atom_array, cutoff = 3.6):
    """
    Filter for potential basepairs based on the distance between the
    nitrogen and oxygen atoms, as potential hydrogen donor/acceptor
    pair.

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
    n_o_mask = (filter_nucleotides(atom_array)
              & np.isin(atom_array.element, ["N", "O"]))
    # Get a boolean mask for atoms that do not belong to the phosphate-
    # backbone
    backbone_mask = (filter_nucleotides(atom_array)
              & ~ np.isin(atom_array.atom_name, ["OP1", "OP2", "OP3", "O5'"]))

    # Combine the N/O-mask with the backbone-mask
    n_o_mask = np.logical_and(n_o_mask, backbone_mask)


    # Get the indices of the N and O atoms that are within the maximum
    # cutoff of each other
    indices = CellList(
        atom_array, cutoff, selection=n_o_mask
    ).get_atoms(atom_array.coord[n_o_mask], cutoff)

    # Loop through the indices of potential partners
    basepair_candidates = {}
    for candidate, partners in zip(np.argwhere(n_o_mask)[:, 0], indices):
        for partner in partners:
            if partner == -1:
                break
            # Find the indices of the first atom of the residues
            candidate_res_start, partner_res_start = get_residue_starts_for(
                atom_array, (candidate, partner)
            )
            # If the basepair candidate is not already in the output
            # list, append to the output list
            if (
                ((partner_res_start, candidate_res_start) \
                not in basepair_candidates)
                and ((candidate_res_start, partner_res_start) \
                not in basepair_candidates)
                and not (candidate_res_start == partner_res_start)
            ):
                basepair_candidates[
                    (candidate_res_start, partner_res_start)
                ] = 1
            elif (candidate_res_start != partner_res_start):
                if (
                    (candidate_res_start, partner_res_start) \
                     in basepair_candidates
                ):
                    basepair_candidates[
                        (candidate_res_start, partner_res_start)
                    ] += 1
                else:
                    basepair_candidates[
                        (partner_res_start, candidate_res_start)
                    ] += 1

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