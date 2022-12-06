# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module allows estimation of secondary structure elements in protein
structures.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["annotate_sse"]

import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack, coord
from .geometry import distance, angle, dihedral
from .filter import filter_amino_acids
from .error import BadStructureError


_r_helix = (np.deg2rad(89-12), np.deg2rad(89+12))
_a_helix = (np.deg2rad(50-20), np.deg2rad(50+20))
_d2_helix = ((5.5-0.5), (5.5+0.5)) # Not used in the algorithm description
_d3_helix = ((5.3-0.5), (5.3+0.5))
_d4_helix = ((6.4-0.6), (6.4+0.6))

_r_strand = (np.deg2rad(124-14), np.deg2rad(124+14))
_a_strand = (np.deg2rad(-180), np.deg2rad(-125),
             np.deg2rad(145), np.deg2rad(180))
_d2_strand = ((6.7-0.6), (6.7+0.6))
_d3_strand = ((9.9-0.9), (9.9+0.9))
_d4_strand = ((12.4-1.1), (12.4+1.1))


def annotate_sse(atom_array, chain_id=None):
    r"""
    Calculate the secondary structure elements (SSE) of a
    peptide chain based on the `P-SEA` algorithm.
    :footcite:`Labesse1997`
    
    The annotation is based CA coordinates only, specifically
    distances and dihedral angles.
    
    Parameters
    ----------
    atom_array : AtomArray
        The atom array to annotate for.
    chain_id : str, optional
        The atoms belonging to this chain are filtered and annotated.
        DEPRECATED: By now multiple chains can be annotated at once.
        To annotate only a certain chain, filter the `atom_array` before
        giving it as input to this function.

    
    Returns
    -------
    sse : ndarray
        An array containing the secondary structure elements,
        where the index corresponds to the index of the CA-filtered
        `atom_array`. 'a' means :math:`{\alpha}`-helix, 'b' means
        :math:`{\beta}`-strand/sheet, 'c' means coil.
    
    Notes
    -----
    Although this function is based on the original `P-SEA` algorithm,
    there are deviations compared to the official `P-SEA` software in
    some cases.
    Do not rely on getting the exact same results.
    
    References
    ----------

    .. footbibliography::
    
    Examples
    --------
    
    SSE of PDB 1L2Y:
    
    >>> sse = annotate_sse(atom_array, "A")
    >>> print(sse)
    ['c' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c'
     'c' 'c']
    
    """
    # Filter all CA atoms in the relevant chain.
    mask = filter_amino_acids(atom_array) & (atom_array.atom_name == "CA")
    if chain_id is not None:
        mask &= atom_array.chain_id == chain_id
    ca_coord = atom_array[mask].coord
    length = len(ca_coord)


    # The distances and angles are not defined for the entire interval,
    # therefore the indices do not have the full range
    # Values that are not defined are NaN
    d2i = np.full(length, np.nan)
    d3i = np.full(length, np.nan)
    d4i = np.full(length, np.nan)
    ri  = np.full(length, np.nan)
    ai  = np.full(length, np.nan)

    d2i[1 : length-1] = distance(ca_coord[0 : length-2], ca_coord[2 : length])
    d3i[1 : length-2] = distance(ca_coord[0 : length-3], ca_coord[3 : length])
    d4i[1 : length-3] = distance(ca_coord[0 : length-4], ca_coord[4 : length])
    ri[1 : length-1]  = angle(
        ca_coord[0 : length-2],
        ca_coord[1 : length-1],
        ca_coord[2 : length]
    )
    ai[1 : length-2] = dihedral(
        ca_coord[0 : length-3],
        ca_coord[1 : length-2],
        ca_coord[2 : length-1],
        ca_coord[3 : length-0]
    )

    
    sse = np.full(length, "c", dtype="U1")
    
    # Find CA that meet criteria for potential helices and strands
    is_potential_helix = (
        (d3i >= _d3_helix[0]) & (d3i <= _d3_helix[1]) &
        (d4i >= _d4_helix[0]) & (d4i <= _d4_helix[1])
    ) | (
        (ri  >= _r_helix[0] ) & ( ri <=  _r_helix[1]) &
        (ai  >= _a_helix[0] ) & ( ai <=  _a_helix[1])
    )
    is_potential_strand = (
        (d2i >= _d2_strand[0]) & (d2i <= _d2_strand[1]) &
        (d3i >= _d3_strand[0]) & (d3i <= _d3_strand[1]) &
        (d4i >= _d4_strand[0]) & (d4i <= _d4_strand[1])
    ) | (
        (ri  >= _r_strand[0] ) & ( ri <=  _r_strand[1]) &
        (
            # Account for periodic boundary of dihedral angle
            ((ai  >= _a_strand[0] ) & ( ai <=  _a_strand[1])) |
            ((ai  >= _a_strand[2] ) & ( ai <=  _a_strand[3]))
        )
    )

    # Real helices are 5 consecutive helix elements
    is_helix = np.zeros(len(sse), dtype=bool)
    counter = 0
    for i in range(len(sse)):
        if is_potential_helix[i]:
            counter += 1
        else:
            if counter >= 5:
                is_helix[i-counter : i] = True
            counter = 0
    # Extend the helices by one at each end if CA meets extension criteria
    i = 0
    while i < length:
        if is_helix[i]:
            sse[i] = "a"
            if (
                d3i[i-1] >= _d3_helix[0] and d3i[i-1] <= _d3_helix[1]
               ) or (
                ri[i-1] >= _r_helix[0] and ri[i-1] <= _r_helix[1]
               ):
                    sse[i-1] = "a"
            sse[i] = "a"
            if (
                d3i[i+1] >= _d3_helix[0] and d3i[i+1] <= _d3_helix[1]
               ) or (
                ri[i+1] >= _r_helix[0] and ri[i+1] <= _r_helix[1]
               ):
                    sse[i+1] = "a"
        i += 1
    
    # Real strands are 5 consecutive strand elements,
    # or shorter fragments of at least 3 consecutive strand residues,
    # if they are in hydrogen bond proximity to 5 other residues
    is_strand = np.zeros(len(sse), dtype=bool)
    counter = 0
    contacts = 0
    for i in range(len(sse)):
        if is_potential_strand[i]:
            counter += 1
            coord = ca_coord[i]
            for strand_coord in ca_coord:
                dist = distance(coord, strand_coord)
                if dist >= 4.2 and dist <= 5.2:
                    contacts += 1
        else:
            if counter >= 4:
                is_strand[i-counter : i] = True
            elif counter == 3 and contacts >= 5:
                is_strand[i-counter : i] = True
            counter = 0
            contacts = 0
    # Extend the strands by one at each end if CA meets extension criteria
    i = 0
    while i < len(sse):
        if is_strand[i]:
            sse[i] = "b"
            if d3i[i-1] >= _d3_strand[0] and d3i[i-1] <= _d3_strand[1]:
                sse[i-1] = "b"
            sse[i] = "b"
            if d3i[i+1] >= _d3_strand[0] and d3i[i+1] <= _d3_strand[1]:
                sse[i+1] = "b"
        i += 1
    
    return sse
            
