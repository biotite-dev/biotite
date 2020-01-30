# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mmtf"
__author__ = "Patrick Kunzmann"
__all__ = ["encode_array"]

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


def encode_array(np.ndarray array, int codec, int param):
    # Pass-through: 32-bit floating-point number array
    if   codec == 1:
        array = array.astype(np.float32, copy=False)
        return array.astype(">f4").tobytes()
    # Pass-through: 8-bit signed integer array
    elif codec == 2:
        array = array.astype(np.int8, copy=False)
        return array.astype(">i1").tobytes()
    # Pass-through: 16-bit signed integer array
    elif codec == 3:
        array = array.astype(np.int16, copy=False)
        return array.astype(">i2").tobytes()
    # Pass-through: 32-bit signed integer array
    elif codec == 4:
        array = array.astype(np.int32, copy=False)
        return array.astype(">i4").tobytes()
    # UTF8/ASCII fixed-length string array
    elif codec == 5:
        dtype = np.dtype("U" + str(param))
        array = array.astype(dtype, copy=False)
        return array.astype(np.dtype("S" + str(param))).tobytes()
    # Run-length encoded character array
    elif codec == 6:
        array = array.astype("U1", copy=False)
        array = _encode_run_length(np.frombuffer(array, dtype=np.int32))
        return array.astype(">i4").tobytes()
    # Run-length encoded 32-bit signed integer array
    elif codec == 7:
        array = array.astype(np.int32, copy=False)
        return _encode_run_length(array).astype(">i4").tobytes()
    # Delta & run-length encoded 32-bit signed integer array
    elif codec == 8:
        array = array.astype(np.int32, copy=False)
        return _encode_run_length(_encode_delta(array)).astype(">i4").tobytes()
    # Integer & run-length encoded 32-bit floating-point number array
    elif codec == 9:
        array = array.astype(np.float32, copy=False)
        return _encode_run_length(
                    _encode_integer(param, array).astype(np.int32)
                ).astype(">i4").tobytes()
    # Integer & delta encoded
    # & two-byte-packed 32-bit floating-point number array
    elif codec == 10:
        array = array.astype(np.float32, copy=False)
        return _encode_packed(
                    True, _encode_delta(
                        _encode_integer(param, array).astype(np.int32)
                    )
                ).astype(">i2").tobytes()
    # Integer encoded 32-bit floating-point number array
    elif codec == 11:
        array = array.astype(np.float32, copy=False)
        return _encode_integer(param, array).astype(">i2").tobytes()
    # Integer & two-byte-packed 32-bit floating-point number array
    elif codec == 12:
        array = array.astype(np.float32, copy=False)
        return _encode_packed(
                    True, _encode_integer(param, array).astype(np.int32) 
                ).astype(">i2").tobytes()
    # Integer & one-byte-packed 32-bit floating-point number array
    elif codec == 13:
        array = array.astype(np.float32, copy=False)
        return _encode_packed(
                    False, _encode_integer(param, array).astype(np.int32)
                ).astype(">i1").tobytes()
    # Two-byte-packed 32-bit signed integer array
    elif codec == 14:
        array = array.astype(np.int32, copy=False)
        return _encode_packed(True, array).astype(">i2").tobytes()
    # One-byte-packed 32-bit signed integer array
    elif codec == 15:
        array = array.astype(np.int32, copy=False)
        return _encode_packed(False, array).astype(">i1").tobytes()
    else:
        raise ValueError(f"Unknown codec with ID {codec}")


def _encode_delta(int32[:] array):
    cdef int32[:] output = np.zeros(array.shape[0], np.int32)
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
            # New element -> Write element with run-length
            output[j] = val
            output[j+1] = run_length
            j += 2
            val = curr_val
            run_length = 1
    # Write last element
    output[j] = val
    output[j+1] = run_length
    j += 2
    # Trim to correct size
    return np.asarray(output)[:j]


@cython.cdivision(True)
def _encode_packed(bint two_byte, int32[:] array):
    cdef int min_val, max_val
    cdef int i=0, j=0
    if two_byte:
        min_val = np.iinfo(np.int16).min
        max_val = np.iinfo(np.int16).max
    else:
        min_val = np.iinfo(np.int8).min
        max_val = np.iinfo(np.int8).max
    # Get length of output array
    # by summing up required length of each element
    cdef int number
    cdef int length = 0
    for i in range(array.shape[0]):
        number = array[i]
        if number < 0:
            length += number // min_val +1
        elif number > 0:
            length += number // max_val +1
        else:
            # e = 0
            length += 1
    # Fill output
    cdef int16[:] output = np.zeros(length, dtype=np.int16)
    cdef int remainder
    j = 0
    for i in range(array.shape[0]):
        remainder = array[i]
        if remainder < 0:
            while remainder <= min_val:
                remainder -= min_val
                output[j] = min_val
                j += 1
        elif remainder > 0:
            while remainder >= max_val:
                remainder -= max_val
                output[j] = max_val
                j += 1
        output[j] = remainder
        j += 1
    return np.asarray(output)


def _encode_integer(int divisor, np.ndarray array):
    return np.multiply(array, divisor)