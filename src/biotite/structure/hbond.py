# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for hydrogen bonding calculation.
"""

__name__ = "biotite.structure"
__author__ = "Daniel Bauer, Patrick Kunzmann"
__all__ = ["hbond", "hbond_frequency"]

import warnings
import numpy as np
from biotite.structure.atoms import AtomArrayStack, stack
from biotite.structure.celllist import CellList
from biotite.structure.geometry import angle, distance


def hbond(
    atoms,
    selection1=None,
    selection2=None,
    selection1_type="both",
    cutoff_dist=2.5,
    cutoff_angle=120,
    donor_elements=("O", "N", "S"),
    acceptor_elements=("O", "N", "S"),
    periodic=False,
):
    r"""
    Find hydrogen bonds in a structure using the Baker-Hubbard
    algorithm. :footcite:`Baker1984`

    This function identifies hydrogen bonds based on the bond angle
    :math:`\theta` and the bond distance :math:`d_{H,A}`.
    The default criteria is :math:`\theta > 120^{\circ}`
    and :math:`d_{H,A} \le 2.5 \mathring{A}`.
    Consequently, the given structure must contain hydrogen atoms.
    Otherwise, no hydrogen bonds will be found.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The atoms to find hydrogen bonds in.
    selection1, selection2 : ndarray, optional
        Boolean mask for atoms to limit the hydrogen bond search to
        specific sections of the model. The shape must match the
        shape of the `atoms` argument. If None is given, the whole atoms
        stack is used instead.
    selection1_type : {'acceptor', 'donor', 'both'}, optional
        Determines the type of `selection1`.
        The type of `selection2` is chosen accordingly
        ('both' or the opposite).
    cutoff_dist : float, optional
        The maximal distance between the hydrogen and acceptor to be
        considered a hydrogen bond.
    cutoff_angle : float, optional
        The angle cutoff in degree between Donor-H..Acceptor to be
        considered a hydrogen bond.
    donor_elements, acceptor_elements : tuple of str
        Elements to be considered as possible donors or acceptors.
    periodic : bool, optional
        If true, hydrogen bonds can also be detected in periodic
        boundary conditions.
        The `box` attribute of `atoms` is required in this case.

    Returns
    -------
    triplets : ndarray, dtype=int, shape=(n,3)
        *n x 3* matrix containing the indices of every Donor-H..Acceptor
        interaction that is available in any of the models.
        *n* is the number of found interactions.
        The three matrix columns are *D_index*, *H_index*, *A_index*.
        If only one model (`AtomArray`) is given, `triplets` contains
        all of its hydrogen bonds.
    mask : ndarry, dtype=bool, shape=(m,n)
        *m x n* matrix that shows if an interaction with index *n* in
        `triplets` is present in the model *m* of the input `atoms`.
        Only returned if `atoms` is an :class:`AtomArrayStack`.

    See Also
    --------
    hbond_frequency : Compute the frequency of each bond over the models.

    Notes
    -----
    The result of this function may include false positives:
    Only the chemical elements and the bond geometry is checked.
    However, there are some cases where a hydrogen bond is still not
    reasonable.
    For example, a nitrogen atom with positive charge could be
    considered as acceptor atom by this method, although this does
    make sense from a chemical perspective.

    References
    ----------

    .. footbibliography::

    Examples
    --------
    Calculate the total number of hydrogen bonds found in each model:

    >>> triplets, mask = hbond(atom_array_stack)
    >>> hbonds_per_model = np.count_nonzero(mask, axis=1)
    >>> print(hbonds_per_model)
    [14 14 14 12 11 12  9 13  9 14 13 13 14 11 11 12 11 14 14 13 14 13 15 17
     14 12 15 12 12 13 13 13 12 12 11 14 10 11]

    Get hydrogen bond donors of third model:

    >>> # Third model -> index 2
    >>> triplets = triplets[mask[2,:]]
    >>> # First column contains donors
    >>> print(atom_array_stack[2, triplets[:,0]])
        A       5  GLN N      N        -5.009   -0.575   -1.365
        A       6  TRP N      N        -2.154   -0.497   -1.588
        A       7  LEU N      N        -1.520   -1.904    0.893
        A       8  LYS N      N        -2.716   -4.413    0.176
        A       8  LYS NZ     N        -6.352   -4.311   -4.482
        A       9  ASP N      N        -0.694   -5.301   -1.644
        A      11  GLY N      N         2.142   -4.244    1.916
        A      10  GLY N      N         1.135   -6.232    0.250
        A      14  SER OG     O         4.689   -5.759   -2.390
        A      13  SER N      N         6.424   -5.220    3.257
        A      14  SER N      N         6.424   -5.506    0.464
        A      15  GLY N      N         8.320   -3.632   -0.318
        A      16  ARG N      N         8.043   -1.206   -1.866
        A       6  TRP NE1    N         3.420    0.332   -0.121
    """
    if not (atoms.element == "H").any():
        warnings.warn(
            "Input structure does not contain hydrogen atoms, "
            "hence no hydrogen bonds can be identified"
        )

    # Create AtomArrayStack from AtomArray
    if not isinstance(atoms, AtomArrayStack):
        atoms = stack([atoms])
        single_model = True
    else:
        single_model = False

    if periodic:
        box = atoms.box
    else:
        box = None

    # Mask for donor/acceptor elements
    donor_element_mask = np.isin(atoms.element, donor_elements)
    acceptor_element_mask = np.isin(atoms.element, acceptor_elements)

    if selection1 is None:
        selection1 = np.ones(atoms.array_length(), dtype=bool)
    if selection2 is None:
        selection2 = np.ones(atoms.array_length(), dtype=bool)

    if selection1_type == "both":
        # The two selections are separated into three selections:
        # the original ones without the overlaping part
        # and one containing the overlap
        # This prevents redundant triplets and unnecessary computation
        overlap_selection = selection1 & selection2
        # Original selections without overlaping part
        exclusive_selection1 = selection1 & (~overlap_selection)
        exclusive_selection2 = selection2 & (~overlap_selection)

        # Put selections to list for cleaner iteration
        selections = [exclusive_selection1, exclusive_selection2, overlap_selection]
        selection_combinations = [
            # (0,0),   is not included, would be same selection
            #         as donor and acceptor simultaneously
            (0, 1),
            (0, 2),
            (1, 0),
            # (1,1),   # same reason above
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),  # overlaping part, combination is necessary
        ]

        all_comb_triplets = []
        all_comb_mask = []
        for selection_index1, selection_index2 in selection_combinations:
            donor_mask = selections[selection_index1]
            acceptor_mask = selections[selection_index2]
            if (
                np.count_nonzero(donor_mask) != 0
                and np.count_nonzero(acceptor_mask) != 0
            ):
                # Calculate triplets and mask
                triplets, mask = _hbond(
                    atoms,
                    donor_mask,
                    acceptor_mask,
                    donor_element_mask,
                    acceptor_element_mask,
                    cutoff_dist,
                    cutoff_angle,
                    box,
                )
                all_comb_triplets.append(triplets)
                all_comb_mask.append(mask)
        # Merge results from all combinations
        triplets = np.concatenate(all_comb_triplets, axis=0)
        mask = np.concatenate(all_comb_mask, axis=1)

    elif selection1_type == "donor":
        triplets, mask = _hbond(
            atoms,
            selection1,
            selection2,
            donor_element_mask,
            acceptor_element_mask,
            cutoff_dist,
            cutoff_angle,
            box,
        )

    elif selection1_type == "acceptor":
        triplets, mask = _hbond(
            atoms,
            selection2,
            selection1,
            donor_element_mask,
            acceptor_element_mask,
            cutoff_dist,
            cutoff_angle,
            box,
        )

    else:
        raise ValueError(f"Unkown selection type '{selection1_type}'")

    if single_model:
        # For a atom array (not stack),
        # hbond_mask contains only 'True' values,
        # since all interaction are in the one model
        # -> Simply return triplets without hbond_mask
        return triplets
    else:
        return triplets, mask


def _hbond(
    atoms,
    donor_mask,
    acceptor_mask,
    donor_element_mask,
    acceptor_element_mask,
    cutoff_dist,
    cutoff_angle,
    box,
):
    # Filter donor/acceptor elements
    donor_mask &= donor_element_mask
    acceptor_mask &= acceptor_element_mask

    first_model_box = box[0] if box is not None else None
    if atoms.bonds is not None:
        donor_h_mask, associated_donor_indices = _get_bonded_h(
            atoms[0], donor_mask, atoms.bonds
        )
    else:
        warnings.warn(
            "Input structure has no associated 'BondList', "
            "Hydrogen atoms bonded to donors are detected by distance"
        )
        donor_h_mask, associated_donor_indices = _get_bonded_h_via_distance(
            atoms[0], donor_mask, first_model_box
        )
    donor_h_i = np.where(donor_h_mask)[0]
    acceptor_i = np.where(acceptor_mask)[0]
    if len(donor_h_i) == 0 or len(acceptor_i) == 0:
        # Return empty triplets and mask
        return (
            np.zeros((0, 3), dtype=int),
            np.zeros((atoms.stack_depth(), 0), dtype=bool),
        )

    # Narrow the amount of possible acceptor to donor-H connections
    # down via the distance cutoff parameter using a cell list
    # Save in acceptor-to-hydrogen matrix
    # (true when distance smaller than cutoff)
    coord = atoms.coord
    possible_bonds = np.zeros((len(acceptor_i), len(donor_h_i)), dtype=bool)
    periodic = False if box is None else True
    for model_i in range(atoms.stack_depth()):
        donor_h_coord = coord[model_i, donor_h_mask]
        acceptor_coord = coord[model_i, acceptor_mask]
        box_for_model = box[model_i] if box is not None else None
        cell_list = CellList(
            donor_h_coord, cell_size=cutoff_dist, periodic=periodic, box=box_for_model
        )
        possible_bonds |= cell_list.get_atoms_in_cells(acceptor_coord, as_mask=True)
    possible_bonds_i = np.where(possible_bonds)
    # Narrow down
    acceptor_i = acceptor_i[possible_bonds_i[0]]
    donor_h_i = donor_h_i[possible_bonds_i[1]]

    # Build D-H..A triplets
    donor_i = associated_donor_indices[donor_h_i]
    triplets = np.stack((donor_i, donor_h_i, acceptor_i), axis=1)
    # Remove entries where donor and acceptor are the same
    triplets = triplets[donor_i != acceptor_i]

    hbond_mask = _is_hbond(
        coord[:, triplets[:, 0]],  # donors
        coord[:, triplets[:, 1]],  # donor hydrogens
        coord[:, triplets[:, 2]],  # acceptors
        box,
        cutoff_dist=cutoff_dist,
        cutoff_angle=cutoff_angle,
    )

    # Reduce output to contain only triplets counted at least once
    is_counted = hbond_mask.any(axis=0)
    triplets = triplets[is_counted]
    hbond_mask = hbond_mask[:, is_counted]

    return triplets, hbond_mask


def _get_bonded_h(array, donor_mask, bonds):
    """
    Helper function to find indices of associated hydrogens in atoms for
    all donors in atoms[donor_mask].
    A `BondsList` is used for detecting bonded hydrogen atoms.
    """
    hydrogen_mask = array.element == "H"

    donor_hydrogen_mask = np.zeros(len(array), dtype=bool)
    associated_donor_indices = np.full(len(array), -1, dtype=int)

    all_bond_indices, _ = bonds.get_all_bonds()
    donor_indices = np.where(donor_mask)[0]

    for donor_i in donor_indices:
        bonded_indices = all_bond_indices[donor_i]
        # Remove padding values
        bonded_indices = bonded_indices[bonded_indices != -1]
        # Filter hydrogen atoms
        bonded_indices = bonded_indices[hydrogen_mask[bonded_indices]]
        donor_hydrogen_mask[bonded_indices] = True
        associated_donor_indices[bonded_indices] = donor_i

    return donor_hydrogen_mask, associated_donor_indices


def _get_bonded_h_via_distance(array, donor_mask, box):
    """
    Helper function to find indices of associated hydrogens in atoms for
    all donors in atoms[donor_mask].
    The criterium is that the hydrogen must be in the same residue and
    the distance must be smaller than the cutoff.
    """
    CUTOFF = 1.5

    coord = array.coord
    res_id = array.res_id
    hydrogen_mask = array.element == "H"

    donor_hydrogen_mask = np.zeros(len(array), dtype=bool)
    associated_donor_indices = np.full(len(array), -1, dtype=int)

    donor_indices = np.where(donor_mask)[0]
    for donor_i in donor_indices:
        candidate_mask = hydrogen_mask & (res_id == res_id[donor_i])
        distances = distance(coord[donor_i], coord[candidate_mask], box=box)
        donor_h_indices = np.where(candidate_mask)[0][distances <= CUTOFF]
        for i in donor_h_indices:
            associated_donor_indices[i] = donor_i
            donor_hydrogen_mask[i] = True

    return donor_hydrogen_mask, associated_donor_indices


def _is_hbond(donor, donor_h, acceptor, box, cutoff_dist, cutoff_angle):
    """
    Filter triplets that meet distance and angle condition.
    """
    cutoff_angle_rad = np.deg2rad(cutoff_angle)
    theta = angle(donor, donor_h, acceptor, box=box)
    dist = distance(donor_h, acceptor, box=box)
    return (theta > cutoff_angle_rad) & (dist <= cutoff_dist)


def hbond_frequency(mask):
    """
    Get the relative frequency of each hydrogen bond in a multi-model
    structure.

    The frequency is the amount of models, where the respective bond
    exists divided by the total amount of models.

    Parameters
    ----------
    mask : ndarray, dtype=bool, shape=(m,n)
        Input mask obtained from `hbond` function.

    Returns
    -------
    ndarray, dtype=Float
        For each individual interaction *n* of the mask, returns the
        percentage of models *m*, in which this hydrogen bond is
        present.

    See Also
    --------
    hbond : Returns the mask that can be input into this function.

    Examples
    --------

    >>> triplets, mask = hbond(atom_array_stack)
    >>> freq = hbond_frequency(mask)
    >>> print(freq)
    [0.263 0.289 0.105 0.105 0.237 0.026 0.053 0.395 1.000 1.000 1.000 0.026
     0.421 0.026 0.026 0.316 0.816 0.026 0.921 0.026 0.342 0.026 0.105 0.026
     0.132 0.053 0.026 0.158 0.026 0.868 0.211 0.026 0.921 0.316 0.079 0.237
     0.105 0.421 0.079 0.026 1.000 0.053 0.132 0.026 0.184]
    """
    return mask.sum(axis=0) / len(mask)
