# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This module allows checking of atom arrays and atom array stacks for errors in the structure.
"""

import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack
from .filter import filter_backbone


def check_id_continuity(array):
    ids = array.res_id
    diff = np.diff(ids)
    discontinuity = np.where( ((diff != 0) & (diff != 1)) )
    return discontinuity[0] + 1

def check_bond_continuity(array, min_len=1.2, max_len=1.8):
    backbone = array[filter_backbone(array)]
    diff = np.diff(backbone.coord, axis=0)
    sq_distance = np.sum(diff**2, axis=1)
    sq_min_len = min_len**2
    sq_max_len = max_len**2
    discontinuity = np.where( ((sq_distance < sq_min_len) & (sq_distance > sq_max_len)) )
    return discontinuity[0] + 1