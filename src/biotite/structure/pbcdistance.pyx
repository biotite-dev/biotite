# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A module for periodic boundary condition aware distance calculation of
triclinic boxes
"""

__author__ = "Patrick Kunzmann"
__all__ = []

cimport cython
cimport numpy as np
from libc.math cimport sqrt

import numpy as np
from .box import fraction_to_coord


def _pair_distance_orthogonal_triclinic(double[:,:] fractions not None,
                                        double[:,:] box not None,
                                        double[:] dist not None):
    cdef int arr_pos
    cdef int i,j,k
    cdef double[:,:] diffs
    cdef double x,y,z
    cdef double min_sq_dist

    if box.shape[0] != 3 or box.shape[1] != 3:
        raise ValueError("Invalid shape for box vectors")
    if fractions.shape[1] != 3:
        raise ValueError("Invalid shape for fraction vectors")
    if dist.shape[0] != fractions.shape[0]:
        raise ValueError(
            f"Distance array has length {dist.shape[0]}, but fractions array "
            f"has length {fractions.shape[0]}"
        )
    diffs = fraction_to_coord(np.asarray(fractions), np.asarray(box))
    # Fraction components are guaranteed to be positive 
    # Test all 3 fraction vector components
    # with positive and negative sign
    # (i,j,k in {-1, 0})
    # Hence, 8 periodic copies are tested
    for arr_pos in range(dist.shape[0]):
        min_sq_dist = _calc_sq_dist(
            diffs[arr_pos,0], diffs[arr_pos,1], diffs[arr_pos,2]
        )
        for i in range(-1, 1):
            for j in range(-1, 1):
                for k in range(-1, 1):
                    x = diffs[arr_pos,0] + i*box[0,0] + j*box[1,0] + k*box[2,0]
                    y = diffs[arr_pos,1] + i*box[0,1] + j*box[1,1] + k*box[2,1]
                    z = diffs[arr_pos,2] + i*box[0,2] + j*box[1,2] + k*box[2,2]
                    min_sq_dist = _min(
                        min_sq_dist,
                        _calc_sq_dist(x,y,z)
                    )
        dist[arr_pos] = sqrt(min_sq_dist)


cdef inline double _min(double val1, double val2):
    return val1 if val1 < val2 else val2

cdef inline double _calc_sq_dist(double x, double y, double z):
    return x*x + y*y + z*z