# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for hydrogen bonding calculation.
"""

__author__ = "Daniel Bauer"
__all__ = ["is_hbond", "hbond"]

from .geometry import distance, angle
import numpy as np
from .atoms import AtomArrayStack, stack


# NOTE Use tuple instead of list as default parameters. See https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments
def hbond(atoms1, atoms2=None, cutoff_dist=2.5, cutoff_angle=120,
           donor_elements=('O', 'N', 'S'), acceptor_elements=('O', 'N', 'S')):
    """
    Finds hydrogen bonds between atoms1 and atoms2.
    
    The default criteria is: :math:`\\theta > 120deg` and :math
    :math:`\\text(H..Acceptor) < 2.5 A` (Baker and Hubbard, 1984)
    
    Parameters
    ----------
    
    atoms1 : AtomArray or AtomArrayStack
        TODO
    atoms2 : AtomArray or AtomArrayStack or None
        TODO
    cutoff_dist: float or int
        The maximal distance between Donor-H..Acceptor to be considered a hydrogen bond
    cutoff_angle: float or int
        The minimal angle between Donor-H..Acceptor to be considered a hydrogen bond  
    donor_elements: array of strings
        Elements to be considered as possible donors
    acceptor_elements: array of strings
        Elements to be considered as possible acceptors
        
    Returns
    -------
    array : int
        Nx3 matrix containing the indices of every Donor-H..Acceptor interaction that was counted at least once. N is the number
        of found interactions. The format is [[D_index, H_intex, A_index]]
    array : bool
        MxN matrix that shows if an interaction N (see above) is present in the model M.
        
    Examples
    --------
    
    Input
    
    >>> struct = load_structure("tests/structure/data/1l2y.pdb")

    >>> triplets, mask = hbond.hbond(struct)
    >>> hbonds_per_model = mask.sum(axis=1)

    >>> plt.plot(range(len(struct)), hbonds_per_model )
    >>> plt.xlabel("Model")
    >>> plt.ylabel("# H-Bonds")
    >>> plt.show()

    
    Output
    
    TODO
        
    

    """

    # Create AtomArrayStacks from AtomArrays
    if not isinstance(atoms1, AtomArrayStack):
        # NOTE: The AtomArrayStack constructor creates an empty stack (similar to np.zeros()). What you want is the function stack()
        atoms1 = stack([atoms1])
    if atoms2 and not isinstance(atoms2, AtomArrayStack):
        atoms2 = stack([atoms2])

    # If no second stack is given, the first one is used
    if not atoms2:
        atoms2 = atoms1

    # atoms1 and atoms 2 need to be of same length
    if atoms1.array_length() != atoms2.array_length():
        raise ValueError("atoms1 and atoms2 must be of same length")

    # Find donors, acceptors and donor hydrogens
    # NOTE: For consistency
    donors = atoms1[:, np.isin(atoms1.element, donor_elements)]
    acceptors = atoms2[:, np.isin(atoms2.element, acceptor_elements)]

    def _get_bonded_hydrogen(atoms1, atoms2, cutoff=1.5):
        """
        Helper function to find associated hydrogens for all donors.
        The criterium is that the hydrogen must be in the same residue
        and the distance must be smaller then 1.5 Angstroem

        """
        hydrogens = atoms2[:, atoms2.element == 'H']

        donor_hs = []
        for i in range(atoms1.array_length()):
            donor = atoms1[0, i]
            candidates = hydrogens[:, hydrogens.res_id == donor.res_id]
            distances = distance(donor, candidates[0])
            donor_h = candidates[:, distances < cutoff]
            donor_hs.append(donor_h)

        return donor_hs

    # TODO use BondList if available
    donor_hs = _get_bonded_hydrogen(donors, atoms1)

    # Build a stack containing the D-H..A triplets in correct order for every possible possible hbond
    # TODO function spends 99.9% of its time in creating the triplets array. how can we make this faster?
    triplets = AtomArrayStack(depth=donors.stack_depth(), length=0)
    for d_i in range(donors.array_length()):
        for a_i in range(acceptors.array_length()):
            for dh_i in range(donor_hs[d_i].array_length()):
                if donors[:, d_i] != acceptors[:, a_i]:
                    triplets += donors[:, d_i] + donor_hs[d_i][:, dh_i] + acceptors[:, a_i]

    # Calculate angle and distance on all triplets
    coords = triplets.coord
    distances = distance(coords[:, 1::3], coords[:, 2::3])
    angles = angle(coords[:, 0::3], coords[:, 1::3], coords[:, 2::3])

    # Apply hbond criterion
    cutoff_angle_radian = np.deg2rad(cutoff_angle)
    hbond_mask = (distances <= cutoff_dist) & (angles >= cutoff_angle_radian)

    # Reduce output to contain only triplets counted at least once
    # NOTE: More clear syntax
    is_counted = hbond_mask.any(axis=0)
    triplets = triplets[:, np.repeat(is_counted, 3)]
    hbond_mask = hbond_mask[:, is_counted]

    return triplets, hbond_mask



def is_hbond(donor, donor_h, acceptor, cutoff_dist=2.5, cutoff_angle=120):
    """
    True if the angle and distance between donor, donor_h and acceptor
    meets the criteria of a hydrogen bond
    
    The default criteria is: :math:`\\theta > 120deg` and :math
    :math:`\\text(H..Acceptor) < 2.5 A` (Baker and Hubbard, 1984)
    """

    cutoff_angle_rad = np.deg2rad(cutoff_angle)
    theta = angle(donor, donor_h, acceptor)
    dist = distance(donor_h, acceptor)

    return theta > cutoff_angle_rad and dist < cutoff_dist

def get_hbond_frequency(mask):
    """
    Parameters
    ----------
    mask: Array
        Input mask obtained from `hbond.hbond`
    
    Returns
    -------
    The frequency for each hydrogen bond

    
    Examples
    --------
        
    >>> struct = load_structure("tests/structure/data/1l2y.pdb")

    >>> triplets, mask = hbond.hbond(struct)
    >>> freq = hbond.get_hbond_frequency(mask)

    See Also
    --------
    hbond
    """
    return mask.sum(axis=0)/len(mask)
