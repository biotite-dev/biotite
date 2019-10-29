# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["find_matches"]

cimport cython
cimport numpy as np
from libc.stdlib cimport realloc, calloc, free

import numpy as np
from ..alphabet import AlphabetError


ctypedef np.int32_t int32
ctypedef np.int64_t int64
ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64

ctypedef fused CodeType:
    uint8
    uint16
    uint32
    uint64


def find_matches(references, query):
    alphabet = query.get_alphabet()
    cdef uint64 alph_length = len(alphabet)
    for i, reference in enumerate(references):
        if reference.get_alphabet() != alphabet:
            raise ValueError(
                f"The alphabet of reference sequence {i} is not equal to the "
                f"query sequence alphabet"
            )
        if not isinstance(reference.code, np.ndarray):
            raise TypeError(
                f"The code of sequence {i} is invalid"
            )
    
    cdef uint64** mapping
    cdef uint64[:] mapping_lengths
    return_val = _create_word_to_pos_mapping(query.code, alph_length)
    # Double casting is necessary to trick compiler
    # to convert integer back into pointer
    mapping = <uint64**> <uint64> (return_val[0])
    mapping_lengths = return_val[1]

    #print()
    #for code in range(alph_length):
    #    m = [mapping[code][i] for i in range(mapping_lengths[code])]
    #    if len(m) != 0:
    #        print(code, m)

    matches = [
        _find_matches(reference.code, <uint64> mapping, mapping_lengths)
        for reference in references
    ]

    _delete_word_to_pos_mapping(mapping, alph_length)
    return matches


# This is a Python function (instead of cdef)
# to be able to infer CodeType at run time
#@cython.boundscheck(False)
#@cython.wraparound(False)
def _find_matches(CodeType[:] reference,
                  # 'query_mapping' is actually a pointer value
                  uint64 query_mapping,
                  uint64[:] mapping_lengths):
    cdef int64 ref_pos, map_pos
    cdef int match_i
    # Convert the as uint64 masquerading pointer into an actual pointer
    cdef uint64** mapping_ptr = <uint64**> query_mapping
    cdef uint64* positions_ptr
    cdef int positions_ptr_length
    cdef CodeType code
    cdef int INIT_SIZE = 1
    cdef uint64[:,:] matches = np.zeros((INIT_SIZE, 2), dtype=np.uint64)

    match_i = 0
    for ref_pos in range(reference.shape[0]):
        code = reference[ref_pos]
        positions_ptr = mapping_ptr[code]
        positions_ptr_length = mapping_lengths[code]
        for map_pos in range(positions_ptr_length):
            if match_i >= matches.shape[0]:
                matches = increase_array_size(np.asarray(matches))
            matches[match_i, 0] = ref_pos
            matches[match_i, 1] = positions_ptr[map_pos]
            match_i += 1

    return np.asarray(matches[:match_i])


cdef np.ndarray increase_array_size(np.ndarray array):
    """
    Double the size of the first dimension of an existing array.
    """
    new_array = np.empty((array.shape[0]*2, array.shape[1]), dtype=array.dtype)
    new_array[:array.shape[0],:] = array
    return new_array


# This is a Python function (instead of cdef)
# to be able to infer CodeType at run time
#@cython.boundscheck(False)
#@cython.wraparound(False)
def _create_word_to_pos_mapping(CodeType[:] sequence not None,
                                uint64 alph_length):
    cdef int64 pos
    cdef uint64 code
    cdef int length
    cdef uint64* positions_ptr
    cdef uint64[:] mapping_lengths = np.zeros(alph_length, dtype=np.uint64)
    cdef uint64** mapping_ptr = <uint64**>calloc(alph_length, sizeof(uint64))
    
    for pos in range(sequence.shape[0]):
        code = sequence[pos]
        length = mapping_lengths[code]
        positions_ptr = mapping_ptr[code]
        positions_ptr = <uint64*>realloc(
            positions_ptr, (length+1) * sizeof(uint64)
        )
        mapping_ptr[code] = positions_ptr
        positions_ptr[length] = pos
        mapping_lengths[code] = length + 1
    
    # Convert pointer to integer to be returnable from a Python function
    return <uint64>mapping_ptr, np.asarray(mapping_lengths)


cdef _delete_word_to_pos_mapping(uint64** mapping, uint64 alph_length):
    cdef uint64 code
    for code in range(alph_length):
        free(mapping[code])
    free(mapping)