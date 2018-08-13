# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for hydrogen bonding calculation.
"""

__author__ = "Daniel Bauer"
__all__ = ["is_hbond", "hbond", "hbond_frequency"]

from .geometry import distance, angle
import numpy as np
from .atoms import AtomArrayStack, stack


def hbond(atoms, donor_selection=None, acceptor_selection=None,
          cutoff_dist=2.5, cutoff_angle=120,
          donor_elements=('O', 'N', 'S'), acceptor_elements=('O', 'N', 'S')):
    """
    Find hydrogen bonds in a structure.
    
    The default criteria is: :math:`\\theta > 120deg` and :math
    :math:`\\text(H..Acceptor) <= 2.5 A` [1]_
    
    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The atoms to find hydrogen bonds in.
    donor_selection, acceptor_selection : ndarray or None
        Boolean mask for atoms to limit the hydrogen bond search to
        specific sections of the model. The shape must match the
        shape of the `atoms` argument (default: None).
    cutoff_dist: float
        The maximal distance between the hydrogen and acceptor to be
        considered a hydrogen bond. (default: 2.5)
    cutoff_angle: float
        The angle cutoff in degree between Donor-H..Acceptor to be
        considered a hydrogen bond (default: 120).
    donor_elements, acceptor_elements: tuple of str
        Elements to be considered as possible donors or acceptors
        (default: O, N, S).
        
    Returns
    -------
    triplets : ndarray, dtype=int, shape=(n,3)
        *n x 3* matrix containing the indices of every Donor-H..Acceptor
        interaction that was counted at least once. *n* is the number of
        found interactions. The format is [[D_index, H_index, A_index]].
    mask : ndarry, dtype=bool, shape=(m,n)
        *m x n* matrix that shows if an interaction with index *n* in
        `triplets` is present in the model *m*.
        
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
    
    .. [1] A Shrake and JA Rupley,
       "Hydrogen bonding in globular proteins"
       Prog Biophys Mol Biol, 44, 97-179 (1984).
    """

    # Create AtomArrayStacks from AtomArrays
    if not isinstance(atoms, AtomArrayStack):
        atoms = stack([atoms])

    # if no donor/acceptor selections are made, use the full stack
    # and reduce selections with multiple models
    if donor_selection is None:
        donor_selection = np.full(atoms.array_length(), True)
    elif len(donor_selection.shape) > 1:
        donor_selection = donor_selection[0]
    if acceptor_selection is None:
        acceptor_selection = np.full(atoms.array_length(), True)
    elif len(acceptor_selection.shape) > 1:
        acceptor_selection = acceptor_selection[0]

    # Filter donor/acceptor elements
    donor_selection = donor_selection & np.isin(atoms.element, donor_elements)
    acceptor_selection = acceptor_selection & np.isin(atoms.element, acceptor_elements)

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
            # print(candidate_mask)
            candidate_distance = distance(donor, atoms[candidate_mask & hydrogens_mask])

            distances = np.full(atoms.array_length(), -1)
            distances[candidate_mask & hydrogens_mask] = candidate_distance
            donor_h_mask = candidate_mask & (distances <= cutoff) & (distances >= 0)
            donor_hs.append(np.where(donor_h_mask)[0])

        return np.array(donor_hs)

    # TODO use BondList if available
    donor_i = np.where(donor_selection)[0]
    acceptor_i = np.where(acceptor_selection)[0]
    donor_hs_i = _get_bonded_hydrogen(atoms[0], donor_selection)

    # Build an index list containing the D-H..A triplets in correct order for every possible possible hbond
    max_triplets_size = len(donor_i) * len(acceptor_i) * max(map(lambda x: len(x), donor_hs_i))
    triplets = np.zeros(max_triplets_size, dtype=np.int64)
    triplet_idx = 0
    for donor_hs_idx, d_i in enumerate(donor_i):
        for a_i in acceptor_i:
            if d_i != a_i:
                for dh_i in donor_hs_i[donor_hs_idx]:
                    triplets[triplet_idx:triplet_idx+3] = (d_i, dh_i, a_i)
                    triplet_idx += 3
    triplets = triplets[:triplet_idx]

    # Calculate angle and distance on all triplets
    coords = atoms[:, triplets].coord
    hbond_mask = is_hbond(coords[:, 0::3], coords[:, 1::3], coords[:, 2::3],
                          cutoff_dist=cutoff_dist, cutoff_angle=cutoff_angle)

    # Reduce+Reshape output to contain only triplets counted at least once
    is_counted = hbond_mask.any(axis=0)
    triplets = triplets[np.repeat(is_counted, 3)]
    triplets = np.reshape(triplets, (int(len(triplets)/3), 3))
    hbond_mask = hbond_mask[:, is_counted]

    return triplets, hbond_mask


def is_hbond(donor, donor_h, acceptor, cutoff_dist=2.5, cutoff_angle=120):
    """
    True if the angle and distance between donor, donor_h and acceptor
    meets the criteria of a hydrogen bond
    
    The default criteria is: :math:`\\theta > 120deg` and :math
    :math:`\\text(H..Acceptor) <= 2.5 A` (Baker and Hubbard, 1984)
    
    Parameters
    ----------
    donor, donor_h, acceptor : ndarray, dtype=float, shape=(MxN) or (N)
        The Coordinates to measure the hydrogen bonding criterium between.
        The three parameters must be of identical shape and either contain
        a list of coordinates (N) or a set of list of coordinates (MxN).
    cutoff_dist: float
        The maximal distance in Angstroem between the hydrogen and acceptor to be
        considered a hydrogen bond. (default: 2.5)
    cutoff_angle: float
        The angle cutoff in degree between Donor-H..Acceptor to be considered a hydrogen
        bond (default: 120).
        
    Returns
    -------
    mask : ndarray, type=bool_, shape=(MxN) or (N)
        For each set of coordinates and dimension, returns a boolean to
        indicate if the coordinates match the hydrogen bonding criterium
        
    
    See Also
    --------
    hbond
    """

    cutoff_angle_rad = np.deg2rad(cutoff_angle)
    theta = angle(donor, donor_h, acceptor)
    dist = distance(donor_h, acceptor)

    return (theta > cutoff_angle_rad) & (dist <= cutoff_dist)


def hbond_frequency(mask):
    """
    Parameters
    ----------
    mask: ndarray, dtype=bool_, shape=(MxN) or (N)
        Input mask obtained from `hbond` function.
    
    Returns
    -------
    ndarray, dtype=Float
        For each individual triplet n of the mask, returns the
        percentage of models M, in which this hydrogen bond is present.

    Examples
    --------
        
    >>> struct = load_structure("tests/structure/data/1l2y.pdb")
    >>> triplets, mask = hbond.hbond(struct)
    >>> freq = hbond.hbond_frequency(mask)


    See Also
    --------
    hbond
    """
    return mask.sum(axis=0)/len(mask)
