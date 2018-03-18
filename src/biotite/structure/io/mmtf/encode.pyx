# Copyright 2018 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

cimport cython
cimport numpy as np

import numpy as np

ctypedef np.int8_t int8
ctypedef np.int16_t int16
ctypedef np.int32_t int32
ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64
ctypedef np.float32_t float32

__all__ = ["encode_array"]


def encode_array(int codec, int param, np.ndarray array):
    cdef bytes raw_bytes
    cdef int param = 0
    cdef int length = 0
    # Pass-through: 32-bit floating-point number array
    if   codec == 1:
        array = np.frombuffer(raw_bytes, dtype=">f4").astype(np.float32)
        return array
    # Pass-through: 8-bit signed integer array
    elif codec == 2:
        array = np.frombuffer(raw_bytes, dtype=">i1").astype(np.int8)
        return array
    # Pass-through: 16-bit signed integer array
    elif codec == 3:
        array = np.frombuffer(raw_bytes, dtype=">i2").astype(np.int16)
        return array
    # Pass-through: 32-bit signed integer array
    elif codec == 4:
        array = np.frombuffer(raw_bytes, dtype=">i4").astype(np.int32)
        return array
    # UTF8/ASCII fixed-length string array
    elif codec == 5:
        array = np.fromstring(raw_bytes, np.dtype("S" + str(param)))
        return array.astype(np.dtype("U" + str(param)))
    # Run-length encoded character array
    elif codec == 6:
        array = np.frombuffer(raw_bytes, dtype=">i4").astype(np.int32)
        return np.frombuffer(_decode_run_length(array), dtype="U1")
    # Run-length encoded 32-bit signed integer array
    elif codec == 7:
        array = np.frombuffer(raw_bytes, dtype=">i4").astype(np.int32)
        return _decode_run_length(array)
    # Delta & run-length encoded 32-bit signed integer array
    elif codec == 8:
        array = np.frombuffer(raw_bytes, dtype=">i4").astype(np.int32)
        return _decode_delta(
               _decode_run_length(array))
    # Integer & run-length encoded 32-bit floating-point number array
    elif codec == 9:
        array = np.frombuffer(raw_bytes, dtype=">i4").astype(np.int32)
        return _decode_integer(param, 
               _decode_run_length(array))
    # Integer & delta encoded
    # & two-byte-packed 32-bit floating-point number array
    elif codec == 10:
        array = np.frombuffer(raw_bytes, dtype=">i2").astype(np.int16)
        return _decode_integer(param, 
               _decode_delta(
               _decode_packed(array)))
    # Integer encoded 32-bit floating-point number array
    elif codec == 11:
        array = np.frombuffer(raw_bytes, dtype=">i2").astype(np.int16)
        return _decode_integer(param, array)
    # Integer & two-byte-packed 32-bit floating-point number array
    elif codec == 12:
        array = np.frombuffer(raw_bytes, dtype=">i2").astype(np.int16)
        return _decode_integer(param, 
               _decode_packed(array))
    # Integer & one-byte-packed 32-bit floating-point number array
    elif codec == 13:
        array = np.frombuffer(raw_bytes, dtype=">i1").astype(np.int8)
        return _decode_integer(param, 
               _decode_packed(array))
    # Two-byte-packed 32-bit signed integer array
    elif codec == 14:
        array = np.frombuffer(raw_bytes, dtype=">i2").astype(np.int16)
        return _decode_packed(array)
    # One-byte-packed 32-bit signed integer array
    elif codec == 15:
        array = np.frombuffer(raw_bytes, dtype=">i1").astype(np.int8)
        return _decode_packed(array)
    else:
        raise ValueError("Unknown codec with ID " + str(codec))


def _encode_delta(int32[:] array):
    cdef int32[:] output = np.zeros(array.shape[0])
    output[0] = array[0]
    cdef int i = 0
    for i in range(1, array.shape[0]):
        output[i] = array[i] - array[i-1]
    return np.asarray(output)


def _encode_run_length(int32[:] array):
    # Pessimistic allocation of output array
    # -> Run length is 1 for every element
    cdef int32[:] output = np.zeros(array.shape[0] * 2, dtype=np.int32)
    cdef int i=0, j=0
    cdef int val = array[0]
    cdef int run_length = 0
    cdef int curr_val
    for i in range(array.shape[0]):
        curr_val = array[i]
        if curr_val == val:
            run_length += 1
        else:
            output[j] = val
            output[j+1] = run_length
            j += 2
            val = curr_val
            run_length = 0
    # Trim to correct size
    return np.asarray(output)[:j]