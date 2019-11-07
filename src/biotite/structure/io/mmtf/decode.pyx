# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mmtf"
__author__ = "Patrick Kunzmann"
__all__ = ["decode_array"]

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


def decode_array(int codec, bytes raw_bytes, int param):
    cdef np.ndarray array
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
        array = np.frombuffer(raw_bytes, np.dtype("S" + str(param)))
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
        raise ValueError("Unknown codec with ID {codec}")


def _decode_delta(np.ndarray array):
    return np.cumsum(array, dtype=np.int32)


def _decode_run_length(int32[:] array):
    cdef int length = 0
    cdef int i, j
    cdef int value, repeat
    # Determine length of output array by summing the run lengths
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


ctypedef fused PackedType:
    int8
    int16
def _decode_packed(PackedType[:] array):
    cdef int min_val, max_val
    if PackedType is int8:
        min_val = np.iinfo(np.int8).min
        max_val = np.iinfo(np.int8).max
    else:
        min_val = np.iinfo(np.int16).min
        max_val = np.iinfo(np.int16).max
    cdef int i, j
    cdef int packed_val, unpacked_val
    # Pessimistic size assumption:
    # The maximum output array length is the input array length
    # in case all values are within the type limits
    cdef int32[:] output = np.zeros(array.shape[0], dtype=np.int32)
    j = 0
    unpacked_val = 0
    for i in range(array.shape[0]):
        packed_val = array[i]
        if packed_val == max_val or packed_val == min_val:
            unpacked_val += packed_val
        else:
            unpacked_val += packed_val
            output[j] = unpacked_val
            unpacked_val = 0
            j += 1
    # Trim to correct size and return
    return np.asarray(output[:j])


def _decode_integer(int divisor, np.ndarray array):
    return np.divide(array, divisor, dtype=np.float32)