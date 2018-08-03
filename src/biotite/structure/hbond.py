# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for geometric measurements between atoms
in a structure, mainly lenghts and angles.
"""

__author__ = "Daniel Bauer"
# __all__ = ["distance", "centroid", "angle", "dihedral", "dihedral_backbone"]

from .geometry import distance, angle
import numpy as np

#### Steps to implement hbonds
# for each frame:
# find possible hbonds based on some criteria
# possible implementations:
    # most naive: go over all possible donor/acceptor pairs and calculate
    # better: save some computation time: find donor/acceptor pairs in correct distances and only calculate angles them


# TODO inline function into hbond method?
def _get_bonded_hydrogen(atoms1, atoms2, cutoff=1.5):
    hydrogens = atoms2[atoms2.element == 'H']

    # TODO vectorize
    donor_hs = []
    for i in range(atoms1.array_length()):
        donor = atoms1[i]
        candidates = hydrogens[hydrogens.res_id == donor.res_id]
        distances = distance(donor, candidates)
        donor_hs.append(candidates[distances < cutoff])

    return donor_hs


def hbonds(donor_struct, acceptor_struct=None, cutoff_dist=2.5, cutoff_angle=120, dist_from_donor=False, donor_elements=['O'], acceptor_elements=['O']):
    """
    TODO donor/acceptor atoms
    
    
    Parameters
    ----------
    structure: AtomArray or AtomArrayStack

    Returns
    -------
    np or pandas?

    """

    donor = donor_struct[np.isin(donor_struct.element,  donor_elements)]
    donor_h = _get_bonded_hydrogen(donor, donor_struct)

    acceptors = acceptor_struct[np.isin(acceptor_struct.element, acceptor_elements)]


    # TODO find acceptor_donor pairs with distance < cutoff
    # TODO calculate Theta for matched distances
    # TODO return format? IDs for each frame?



def is_hbond(donor, donor_h, acceptor, cutoff_dist=2.5, cutoff_angle=120):
    """ True if the angle and distance between donor, donor_h and acceptor
    meets the criteria of a hydrogen bond
    
    The default criteria is: :math:`\\theta > 120deg` and :math
    :math:`\\text(H..Acceptor) < 2.5 A` (Baker and Hubbard, 1984)
    """

    cutoff_angle_rad = np.deg2rad(cutoff_angle)
    theta = angle(donor, donor_h, acceptor)
    dist = distance(donor_h, acceptor)

    return theta < cutoff_angle_rad and dist < cutoff_dist