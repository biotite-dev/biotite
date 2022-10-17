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


_radians_to_angle = 2*np.pi/360

_r_helix = ((89-12)*_radians_to_angle, (89+12)*_radians_to_angle)
_a_helix = ((50-20)*_radians_to_angle, (50+20)*_radians_to_angle)
_d2_helix = ((5.5-0.5), (5.5+0.5))
_d3_helix = ((5.3-0.5), (5.3+0.5))
_d4_helix = ((6.4-0.6), (6.4+0.6))

_r_strand = ((124-14)*_radians_to_angle, (124+14)*_radians_to_angle)
_a_strand = ((-180)*_radians_to_angle, (-125)*_radians_to_angle,
             (145)*_radians_to_angle, (180)*_radians_to_angle)
_d2_strand = ((6.7-0.6), (6.7+0.6))
_d3_strand = ((9.9-0.9), (9.9+0.9))
_d4_strand = ((12.4-1.1), (12.4+1.1))


def annotate_sse(atom_array, chain_id):
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
    chain_id : str
        The chain ID to annotate for.
    
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
    there are deviatons compared to the official `P-SEA` software in
    some cases. Do not rely on getting the exact same results.
    
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
    ca_coord = atom_array[filter_amino_acids(atom_array) &
                          (atom_array.atom_name == "CA") &
                          (atom_array.chain_id == chain_id)].coord
    
    d2i_coord = np.full(( len(ca_coord), 2, 3 ), np.nan)
    d3i_coord = np.full(( len(ca_coord), 2, 3 ), np.nan)
    d4i_coord = np.full(( len(ca_coord), 2, 3 ), np.nan)
    ri_coord = np.full(( len(ca_coord), 3, 3 ), np.nan)
    ai_coord = np.full(( len(ca_coord), 4, 3 ), np.nan)
    
    # The distances and angles are not defined for the entire interval,
    # therefore the indices do not have the full range
    # Values that are not defined are NaN
    for i in range(1, len(ca_coord)-1):
        d2i_coord[i] = (ca_coord[i-1], ca_coord[i+1])
    for i in range(1, len(ca_coord)-2):
        d3i_coord[i] = (ca_coord[i-1], ca_coord[i+2])
    for i in range(1, len(ca_coord)-3):
        d4i_coord[i] = (ca_coord[i-1], ca_coord[i+3])
    for i in range(1, len(ca_coord)-1):
        ri_coord[i] = (ca_coord[i-1], ca_coord[i], ca_coord[i+1])
    for i in range(1, len(ca_coord)-2):
        ai_coord[i] = (ca_coord[i-1], ca_coord[i],
                       ca_coord[i+1], ca_coord[i+2])
    
    d2i = distance(d2i_coord[:,0], d2i_coord[:,1])
    d3i = distance(d3i_coord[:,0], d3i_coord[:,1])
    d4i = distance(d4i_coord[:,0], d4i_coord[:,1])
    ri = angle(ri_coord[:,0], ri_coord[:,1], ri_coord[:,2])
    ai = dihedral(ai_coord[:,0], ai_coord[:,1],
                  ai_coord[:,2], ai_coord[:,3])
    
    sse = np.full(len(ca_coord), "c", dtype="U1")
    
    # Annotate helices
    # Find CA that meet criteria for potential helices
    is_pot_helix = np.zeros(len(sse), dtype=bool)
    for i in range(len(sse)):
        if (
                d3i[i] >= _d3_helix[0] and d3i[i] <= _d3_helix[1]
            and d4i[i] >= _d4_helix[0] and d4i[i] <= _d4_helix[1]
           ) or (
                ri[i] >= _r_helix[0] and ri[i] <= _r_helix[1]
            and ai[i] >= _a_helix[0] and ai[i] <= _a_helix[1]
           ):
                is_pot_helix[i] = True
    # Real helices are 5 consecutive helix elements
    is_helix = np.zeros(len(sse), dtype=bool)
    counter = 0
    for i in range(len(sse)):
        if is_pot_helix[i]:
            counter += 1
        else:
            if counter >= 5:
                is_helix[i-counter : i] = True
            counter = 0
    # Extend the helices by one at each end if CA meets extension criteria
    i = 0
    while i < len(sse):
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
    
    # Annotate sheets
    # Find CA that meet criteria for potential strands
    is_pot_strand = np.zeros(len(sse), dtype=bool)
    for i in range(len(sse)):
        if (    d2i[i] >= _d2_strand[0] and d2i[i] <= _d2_strand[1]
            and d3i[i] >= _d3_strand[0] and d3i[i] <= _d3_strand[1]
            and d4i[i] >= _d4_strand[0] and d4i[i] <= _d4_strand[1]
           ) or (
                ri[i] >= _r_strand[0] and ri[i] <= _r_strand[1]
            and (   (ai[i] >= _a_strand[0] and ai[i] <= _a_strand[1])
                 or (ai[i] >= _a_strand[2] and ai[i] <= _a_strand[3]))
           ):
                is_pot_strand[i] = True
    # Real strands are 5 consecutive strand elements,
    # or shorter fragments of at least 3 consecutive strand residues,
    # if they are in hydrogen bond proximity to 5 other residues
    pot_strand_coord = ca_coord[is_pot_strand]
    is_strand = np.zeros(len(sse), dtype=bool)
    counter = 0
    contacts = 0
    for i in range(len(sse)):
        if is_pot_strand[i]:
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
            
