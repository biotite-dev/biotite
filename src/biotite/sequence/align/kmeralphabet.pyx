# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["KmerAlphabet"]

cimport cython
cimport numpy as np

import numpy as np
from ..alphabet import Alphabet, AlphabetError


ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64
ctypedef np.int64_t int64


ctypedef fused CodeType:
    uint8
    uint16
    uint32
    uint64


class KmerAlphabet(Alphabet):
    
    def __init__(self, base_alphabet, k):
        if not isinstance(base_alphabet, Alphabet):
            raise TypeError(
                f"Got {type(base_alphabet).__name__}, "
                f"but Alphabet was expected"
            )
        if k < 2:
            raise ValueError("k must be at least 2")
        self._base_alph = base_alphabet
        self._k = k
        base_alph_len = len(self._base_alph)
        self._radix_multiplier = np.array(
            [base_alph_len**n for n in range(self._k)],
            dtype=np.int64
        )
    

    @property
    def k(self):
        return self._k
    
    @property
    def base_alphabet(self):
        return self._base_alph
    

    def get_symbols(self):
        return [self.decode(code) for code in range(len(self))]
    

    def extends(self, alphabet):
        # A KmerAlphabet cannot really extend another KmerAlphabet:
        # If k is not equal, all symbols are not equal
        # If the base alphabet has additional symbols, the correct
        # order is not preserved
        # The only way that KmerAlphabet 'extends' another KmerAlphabet,
        # if the two alphabets are equal
        return alphabet == self
    

    def encode(self, symbol):
        return self.fuse(self._base_alph.encode_multiple(symbol))
    

    def decode(self, code):
        return self._base_alph.decode_multiple(self.split(code))
    

    def fuse(self, codes):
        if codes.shape[-1] != self._k:
            raise AlphabetError(
                f"Given k-mer(s) has {codes.shape[-1]} symbols, "
                f"but alphabet expects {self._k}-mers"
            )
        if np.any(codes > len(self._base_alph)):
            raise AlphabetError("Given k-mer(s) contains invalid symbol code")
        
        orig_shape = codes.shape
        codes = np.atleast_2d(codes)
        kmer_code = np.sum(self._radix_multiplier * codes, axis=-1)
        # The last dimension is removed since it collpased in np.sum
        return kmer_code.reshape(orig_shape[:-1])
    
    def split(self, code):
        if np.any(code >= len(self)) or np.any(code < 0):
            raise AlphabetError(
                f"Given k-mer symbol code is invalid for this alphabet"
            )
        
        orig_shape = np.shape(code)
        split_codes = self._split(np.atleast_1d(code))
        return split_codes.reshape(orig_shape + (self._k,))

    
    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    def _split(self, int64[:] codes not None):
        cdef int i, n
        cdef int64 code, val, symbol_code

        cdef int64[:] radix_multiplier = self._radix_multiplier

        cdef int64[:,:] split_codes = np.empty(
            (codes.shape[0], self._k), dtype=np.int64
        )
        
        for i in range(codes.shape[0]):
            code = codes[i]
            for n in reversed(range(self._k)):
                val = radix_multiplier[n]
                symbol_code = code // val
                split_codes[i,n] = symbol_code
                code -= symbol_code * val
        
        return np.asarray(split_codes)
    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def create_kmers(self, CodeType[:] seq_code not None):
        """
        create_kmers(seq_code)
        """
        cdef int64 i
        cdef int j

        cdef int k = self._k
        cdef uint64 alphabet_length = len(self._base_alph)
        cdef int64[:] radix_multiplier = self._radix_multiplier

        cdef int64[:] kmers = np.empty(len(seq_code) - k + 1, dtype=np.int64)
        
        cdef CodeType code
        cdef int64 kmer
        for i in range(kmers.shape[0]):
            kmer = 0
            for j in range(k):
                code = seq_code[i+j]
                if code >= alphabet_length:
                    raise AlphabetError(f"Symbol code {code} is out of range")
                kmer += radix_multiplier[j] * code
            kmers[i] = kmer
        
        return kmers
    

    def __str__(self):
        return str(self.get_symbols())
    

    def __eq__(self, item):
        if item is self:
            return True
        if not isinstance(item, KmerAlphabet):
            return False
        if self._base_alph != item._base_alph:
            return False
        if self._k != item._k:
            return False
        return True
    

    def __len__(self):
        return int(len(self._base_alph) ** self._k)