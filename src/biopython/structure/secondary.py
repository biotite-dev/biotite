# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This module allows estimation of secondary structure elements in protein
structures.
"""

import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack, coord
from .geometry import distance, angle, dihedral
from .filter import filter_amino_acids
from .error import BadStructureError

__all__ = ["secondary_structure"]


_radians_to_angle = 2*np.pi/360

_r_helix = ((89-12)*_radians_to_angle, (89+12)*_radians_to_angle)
_a_helix = ((50-20)*_radians_to_angle, (50+20)*_radians_to_angle)
_d2_helix = ((5.5-0.5), (5.5+0.5))
_d3_helix = ((5.3-0.5), (5.3+0.5))
_d4_helix = ((6.4-0.6), (6.4+0.6))

_r_strand = ((124-14)*_radians_to_angle, (124+14)*_radians_to_angle)
_a_strand = ((-170-45)*_radians_to_angle, (-170+45)*_radians_to_angle)
_d2_strand = ((6.7-0.6), (6.7+0.6))
_d3_strand = ((9.9-0.9), (9.9+0.9))
_d4_strand = ((12.4-1.1), (12.4+1.1))


def secondary_structure(atom_array, chain_id):
    """
    Calculate the secondary structure of a peptide chain using the
    `P-SEA` algorithm [1]_.
    
    References
    ----------
    
    .. [1] G Labesse, N Colloch, J Pothier, JP Mornon
       "P-SEA: a new efficient assignment of secondary structure from
       CA trace of protein."
       Comput Appl Biosci, 13, 291-295 (1997).
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
    # Find CA that meet criteria for helix starts
    is_helix_start = np.zeros(len(sse), dtype=bool)
    for i in range(len(sse)):
        if (
                d3i[i] >= _d3_helix[0] and d3i[i] <= _d3_helix[1]
            and d4i[i] >= _d4_helix[0] and d4i[i] <= _d4_helix[1]
           ) or (
                ri[i] >= _r_helix[0] and ri[i] <= _r_helix[1]
            and ai[i] >= _a_helix[0] and ai[i] <= _a_helix[1]
           ):
                is_helix_start[i] = True
    # Remove potential helix starts, if there are not at least
    # 5 consecutive elements
    is_real_helix_start = np.zeros(len(sse), dtype=bool)
    counter = 0
    for i in range(len(sse)):
        if is_helix_start[i]:
            counter += 1
        else:
            if counter >= 5:
                is_real_helix_start[i-counter : i] = True
            counter = 0
    # Extend the helices by one at each end if CA meets extension criteria
    
    i = 0
    while i < len(sse):
        if is_real_helix_start[i]:
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
    
    
    #sse[is_real_helix_start] = "a"
    return sse
            
