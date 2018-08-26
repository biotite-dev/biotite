# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for hydrogen bonding calculation.
"""

__author__ = "Daniel Bauer, Patrick Kunzmann"
__all__ = ["hbond", "hbond_frequency"]

from .geometry import distance, angle
import numpy as np
from .atoms import AtomArrayStack, stack


def hbond(atoms, selection1=None, selection2=None, selection1_type='both',
          cutoff_dist=2.5, cutoff_angle=120,
          donor_elements=('O', 'N', 'S'), acceptor_elements=('O', 'N', 'S'),
          vectorized=True):
    """
    Find hydrogen bonds in a structure.
    
    The default criteria is: :math:`\\theta > 120deg` and :math
    :math:`\\text(H..Acceptor) <= 2.5 A` [1]_
    
    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The atoms to find hydrogen bonds in.
    selection1, selection2: ndarray or None
        Boolean mask for atoms to limit the hydrogen bond search to
        specific sections of the model. The shape must match the
        shape of the `atoms` argument. If None is given, the whole atoms
        stack is used instead. (default: None).
    selection1_type: str (default: 'both')
        Can be 'acceptor', 'donor' or 'both' and determines the type of
        selection1. selection2_type is chosen accordingly (both or the
        opposite)
    cutoff_dist: float
        The maximal distance between the hydrogen and acceptor to be
        considered a hydrogen bond. (default: 2.5)
    cutoff_angle: float
        The angle cutoff in degree between Donor-H..Acceptor to be
        considered a hydrogen bond (default: 120).
    donor_elements, acceptor_elements: tuple of str
        Elements to be considered as possible donors or acceptors
        (default: O, N, S).
    vectorized: bool
        Enable/Disable vectorization across models. Vectorization is
        faster, but requires more memory (default: True).
        
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
        Only returned if `atoms` is an `AtomArrayStack`.
        
    Examples
    --------
    Calculate the total number of hydrogen bonds found in each model:
    
    >>> stack = load_structure("path/to/1l2y.pdb")
    >>> triplets, mask = hbond(stack)
    >>> hbonds_per_model = np.count_nonzero(mask, axis=1)
    >>> print(hbonds_per_model)
    [14 15 15 13 11 13  9 14  9 15 13 13 15 11 11 13 11 14 14 13 14 13 15 17
     14 12 15 12 12 13 13 13 12 12 11 15 10 11]

    Get hydrogen bond donors of third model:

    >>> # Third model -> index 2
    >>> triplets = triplets[mask[2,:]]
    >>> # First column contains donors
    >>> print(stack[2, triplets[:,0]])
        A       1 ASN N      N        -6.589    7.754   -0.571
        A       5 GLN N      N        -5.009   -0.575   -1.365
        A       6 TRP N      N        -2.154   -0.497   -1.588
        A       6 TRP NE1    N         3.420    0.332   -0.121
        A       7 LEU N      N        -1.520   -1.904    0.893
        A       8 LYS N      N        -2.716   -4.413    0.176
        A       8 LYS NZ     N        -6.352   -4.311   -4.482
        A       9 ASP N      N        -0.694   -5.301   -1.644
        A      10 GLY N      N         1.135   -6.232    0.250
        A      11 GLY N      N         2.142   -4.244    1.916
        A      13 SER N      N         6.424   -5.220    3.257
        A      14 SER N      N         6.424   -5.506    0.464
        A      14 SER OG     O         4.689   -5.759   -2.390
        A      15 GLY N      N         8.320   -3.632   -0.318
        A      16 ARG N      N         8.043   -1.206   -1.866

    See Also
    --------
    hbond_frequency
    is_hbond

    References
    ----------
    
    .. [1] EN Baker and RE Hubbard,
       "Hydrogen bonding in globular proteins"
       Prog Biophys Mol Biol, 44, 97-179 (1984).
    """

    # Create AtomArrayStack from AtomArray
    if not isinstance(atoms, AtomArrayStack):
        atoms = stack([atoms])
        single_model = True
    else:
        single_model = False

    # determine selection2 type
    if selection1_type == 'both':
        selection2_type = selection1_type
    elif selection1_type == 'acceptor':
        selection2_type = 'donor'
    else:
        selection2_type = 'acceptor'

    # create donors and acceptors selections
    def build_donor_acceptor_selections(selection, selection_type):
        if selection is None:
            selection = np.full(atoms.array_length(), True)

        if selection_type in ['both', 'donor']:
            donor_selection = selection
        else:
            donor_selection = np.full(atoms.array_length(), False)

        if selection_type in ['both', 'acceptor']:
            acceptor_selection = selection
        else:
            acceptor_selection = np.full(atoms.array_length(), False)
        return donor_selection, acceptor_selection


    donor1_selection, acceptor1_selection = \
        build_donor_acceptor_selections(selection1, selection1_type)
    donor2_selection, acceptor2_selection = \
        build_donor_acceptor_selections(selection2, selection2_type)

    # find hydrogen bonds between selections.
    triplets, mask = _hbond(atoms, donor1_selection, acceptor2_selection,
                            cutoff_dist, cutoff_angle, donor_elements,
                            acceptor_elements,
                            vectorized)

    # If the selections are identical, we can skip the second run
    if not (np.array_equal(donor1_selection, donor2_selection) and
            np.array_equal(acceptor1_selection, acceptor2_selection)):
        triplets2, mask2 = _hbond(atoms, donor2_selection, acceptor1_selection,
                                cutoff_dist, cutoff_angle, donor_elements,
                                acceptor_elements,
                                vectorized)
        triplets = np.concatenate((triplets, triplets2))
        mask = np.concatenate((mask, mask2), axis=1)

    if single_model:
        # For a single model, hbond_mask contains only 'True' values,
        # since all interaction are in the one model
        # -> Simply return triplets without hbond_mask
        return triplets
    else:
        return triplets, mask




def _hbond(atoms, donor_selection, acceptor_selection,
           cutoff_dist, cutoff_angle, donor_elements, acceptor_elements,
           vectorized):
    """
    Find hydrogen bonds between the donors and acceptors in a structure.
    
    See Also
    --------
    hbond

    """
    # Filter donor/acceptor elements
    donor_selection \
        = donor_selection & np.isin(atoms.element, donor_elements)
    acceptor_selection \
        = acceptor_selection & np.isin(atoms.element, acceptor_elements)

    def _get_bonded_hydrogen(atoms, donor_mask, cutoff=1.5):
        """
        Helper function to find indeces of associated hydrogens in atoms for
        all donors in atoms[donor_mask]. The criterium is that the hydrogen
        must be in the same residue and the distance must be smaller then 1.5
        Angstroem.

        """
        hydrogens_mask = atoms.element == 'H'
        donors = atoms[donor_mask]
        donor_hs = []
        for i in range(donors.array_length()):
            donor = donors[i]
            candidate_mask = hydrogens_mask & (atoms.res_id == donor.res_id)
            candidate_distance = distance(
                donor, atoms[candidate_mask & hydrogens_mask]
            )

            distances = np.full(atoms.array_length(), -1)
            distances[candidate_mask & hydrogens_mask] = candidate_distance
            donor_h_mask \
                = candidate_mask & (distances <= cutoff) & (distances >= 0)
            donor_hs.append(np.where(donor_h_mask)[0])

        return np.array(donor_hs)

    # TODO use BondList if available
    donor_i = np.where(donor_selection)[0]
    acceptor_i = np.where(acceptor_selection)[0]
    donor_hs_i = _get_bonded_hydrogen(atoms[0], donor_selection)

    # Build an index list containing the D-H..A triplets
    # in correct order for every possible possible hbond
    # The size of the list is 3 times the worst case amount of triplets
    if len(donor_i) == 0:
        max_triplets_size = 0
    else:
        max_triplets_size \
            = 3 * len(donor_i) * len(acceptor_i) \
               * max(map(lambda x: len(x), donor_hs_i))\

    triplets = np.zeros(max_triplets_size, dtype=np.int64)
    triplet_idx = 0
    for donor_hs_idx, d_i in enumerate(donor_i):
        for a_i in acceptor_i:
            if d_i != a_i:
                for dh_i in donor_hs_i[donor_hs_idx]:
                    triplets[triplet_idx:triplet_idx+3] = (d_i, dh_i, a_i)
                    triplet_idx += 3
    triplets = triplets[:triplet_idx]

    if vectorized:
        coords = atoms[:, triplets].coord
        hbond_mask = _is_hbond(coords[:, 0::3], coords[:, 1::3], coords[:, 2::3],
                  cutoff_dist=cutoff_dist, cutoff_angle=cutoff_angle)
    else:
        # calculate mask along the trajectory. Vectorization along axis=0 requires too much memory
        hbond_mask = np.full((len(atoms), int(len(triplets)/3)), False)
        for frame in range(len(atoms)):
            # Calculate angle and distance on all triplets
            coords = atoms[frame, triplets].coord
            frame_mask = _is_hbond(coords[0::3], coords[1::3], coords[2::3],
                               cutoff_dist=cutoff_dist, cutoff_angle=cutoff_angle)
            hbond_mask[frame] = frame_mask

    # Reduce+Reshape output to contain only triplets counted at least once
    is_counted = hbond_mask.any(axis=0)
    triplets = triplets[np.repeat(is_counted, 3)]
    triplets = np.reshape(triplets, (int(len(triplets)/3), 3))
    hbond_mask = hbond_mask[:, is_counted]

    return triplets, hbond_mask


def hbond_frequency(mask):
    """
    Parameters
    ----------
    mask: ndarray, dtype=bool, shape=(m,n)
        Input mask obtained from `hbond` function.
    
    Returns
    -------
    ndarray, dtype=Float
        For each individual interaction *n* of the mask, returns the
        percentage of models *m*, in which this hydrogen bond is
        present.

    See Also
    --------
    hbond

    Examples
    --------

    >>> stack = load_structure("path/to/1l2y.pdb")
    >>> triplets, mask = hbond(stack)
    >>> freq = hbond_frequency(mask)
    >>> print(freq)
    [0.10526316 0.23684211 0.02631579 0.23684211 0.05263158 0.26315789
     0.02631579 0.28947368 0.10526316 0.39473684 1.         1.
     1.         1.         0.02631579 0.02631579 0.02631579 0.02631579
     0.02631579 0.42105263 0.31578947 0.92105263 0.81578947 0.86842105
     0.02631579 0.21052632 0.10526316 0.92105263 0.07894737 0.02631579
     0.34210526 0.10526316 0.02631579 0.31578947 0.23684211 0.42105263
     0.13157895 0.07894737 0.02631579 0.05263158 0.02631579 0.15789474
     0.02631579 0.05263158 0.13157895 0.18421053]
    """
    return mask.sum(axis=0)/len(mask)


def _is_hbond(donor, donor_h, acceptor, cutoff_dist=2.5, cutoff_angle=120):
    """
    True if the angle and distance between donor, donor_h and acceptor
    meets the criteria of a hydrogen bond
    
    The default criteria is: :math:`\\theta > 120deg` and :math
    :math:`\\text(H..Acceptor) <= 2.5 A` (Baker and Hubbard, 1984)
    
    Parameters
    ----------
    donor, donor_h, acceptor : ndarray, dtype=float, shape=(MxN) or (N)
        The coordinates to measure the hydrogen bonding criterium
        between.
        The three parameters must be of identical shape and either
        contain a list of coordinates (N) or a set of list of
        coordinates (MxN).
    cutoff_dist: float
        The maximal distance between the hydrogen and acceptor to be
        considered a hydrogen bond. (default: 2.5)
    cutoff_angle: float
        The angle cutoff in degree between Donor-H..Acceptor to be
        considered a hydrogen bond (default: 120).
        
    Returns
    -------
    mask : ndarray, type=bool_, shape=(MxN) or (N)
        For each set of coordinates and dimension, returns a boolean to
        indicate if the coordinates match the hydrogen bonding
        criterium.
        
    See Also
    --------
    hbond
    """
    cutoff_angle_rad = np.deg2rad(cutoff_angle)
    theta = angle(donor, donor_h, acceptor)
    dist = distance(donor_h, acceptor)

    return (theta > cutoff_angle_rad) & (dist <= cutoff_dist)
