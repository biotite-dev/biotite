# Copyright 2018 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

cimport cython
cimport numpy as np

import numpy as np

ctypedef np.int32_t int32
ctypedef np.int64_t int64
ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64

__all__ = ["decode_array"]


def decode_array(int codec, bytes raw_bytes):
    cdef np.ndarray array
    if   codec == 8:
             array = np.frombuffer(raw_bytes, dtype=">i4").astype(np.int32)
             return _delta_decode(
                    _run_length_decode(array))
    else:
        raise ValueError("Unknown codec with ID " + str(codec))


def _delta_decode(int32[:] array):
    return np.asarray(array).cumsum()


def _run_length_decode(int32[:] array):
    cdef int length = 0
    cdef int i, j
    # Determine length of output array
    for i in range(1, array.shape[0], 2):
        length += array[i]
    cdef int32[:] output = np.zeros(length, dtype=np.int32)
    # Fill output array
    j = 0
    for i in range(0, array.shape[0], 2):
        value = array[i]
        repeat = array[i+1]
        output[j : j+repeat] = value
        j += repeat
    return np.asarray(output)