# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["combine_into_words"]

cimport cython
cimport numpy as np

import numpy as np
from ..alphabet import AlphabetError


ctypedef np.int32_t int32
ctypedef np.int64_t int64
ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64

ctypedef fused CodeType1:
    uint8
    uint16
    uint32
    uint64
ctypedef fused CodeType2:
    uint8
    uint16
    uint32
    uint64


@cython.boundscheck(False)
@cython.wraparound(False)
def combine_into_words(CodeType1[:] base_code not None,
                       CodeType2[:] word_code not None,
                       int word_length, uint64 base_alph_length):
    cdef int64 i
    cdef int j
    
    if word_length < 2:
        raise ValueError("Word length must be at least 2")
    if base_alph_length < 1:
        raise ValueError("Base alphabet length must be at least 1")
    if word_code.shape[0] != base_code.shape[0] - word_length + 1:
        raise ValueError(
            "Length of word code array does not fit length of base code array"
        )
    
    cdef uint64[:] radix_multiplier = np.array(
        [base_alph_length**n for n in range(word_length)],
        dtype=np.uint64
    )
    
    cdef CodeType1 base
    cdef CodeType2 word
    for i in range(word_code.shape[0]):
        word = 0
        for j in range(word_length):
            base = base_code[i+j]
            if base >= base_alph_length:
                raise AlphabetError(f"Symbol code {base} is out of range")
            word += radix_multiplier[word_length - j - 1] * base
        word_code[i] = word