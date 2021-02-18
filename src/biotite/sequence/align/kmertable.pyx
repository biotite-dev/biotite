# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["KmerTable"]

cimport cython
cimport numpy as np
from libc.stdlib cimport realloc, malloc, free

import numpy as np
from ..alphabet import LetterAlphabet
from .kmeralphabet import KmerAlphabet


ctypedef np.int32_t int32
ctypedef np.int64_t int64
ctypedef np.uint32_t uint32
ctypedef np.uint64_t ptr


cdef class KmerTable:

    cdef object _kmer_alph
    cdef int _k

    # The pointer array is the core of the index table:
    # It maps each possible k-mer (represented by its code) to a
    # C-array of indices.
    # Each entry in a C-array points to a reference sequence and the
    # location in that sequence where the respective k-mer appears
    # The memory layout of each C-array is as following:
    #
    # (Array length) (RefIndex 0) (Position 0) (RefIndex 1) (Position 1) ...
    # -----int64----|---uint32---|---uint32---|---uint32---|---uint32---
    #
    # If there is no entry for a k-mer, the respective pointer is NULL.
    cdef ptr[:] _ptr_array


    def __cinit__(self, alphabet, k):
        # This check is necessary for proper memory management
        # of the allocated arrays
        if self._is_initialized():
            raise Exception("Duplicate call of constructor")
        
        if k < 2:
            raise ValueError("k must be at least 2")
        if len(alphabet) < 1:
            raise ValueError("Alphabet length must be at least 1")
        
        self._kmer_alph = KmerAlphabet(alphabet, k)
        self._k = k
        self._ptr_array = np.zeros(len(self._kmer_alph), dtype=np.uint64)


    @property
    def kmer_alphabet(self):
        return self._kmer_alph
    
    @property
    def alphabet(self):
        return self._kmer_alph.base_alphabet
    
    @property
    def k(self):
        return self._k

    
    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    def add(self, sequence, uint32 reference_index, mask=None):
        """
        If this function raises a :class:`MemoryError` the
        :class:`KmerTable` becomes invalid.
        """
        cdef int i
        cdef uint32 seq_pos
        cdef int64 length
        cdef int64 kmer
        cdef uint32* kmer_ptr

        # Store in new variables
        # to disable repetitive initialization checks
        cdef ptr[:] ptr_array = self._ptr_array
        cdef int n_kmers = len(self._kmer_alph)

        code = sequence.code
        
        if len(code) < self._k:
            raise ValueError("Sequence code is shorter than k")
        if self._kmer_alph.base_alphabet != sequence.alphabet:
            raise ValueError(
                "The alphabet used for the k-mer index table is not equal to "
                "the alphabet of the sequence"
            )
        
        cdef int64[:] kmers = self._kmer_alph.create_kmers(code)

        for seq_pos in range(kmers.shape[0]):
            kmer = kmers[seq_pos]
            kmer_ptr = <uint32*> ptr_array[kmer]
            # Calculate new length (i.e. number of entries)
            if kmer_ptr == NULL:
                # C-array does not exist yet
                # Initial C-array contains 2 elements:
                # 1. int64 array length
                # 2. 2 x uint32 for first entry
                length = 2 + 2
            else:
                length = (<int64*> kmer_ptr)[0] + 2

            # Reallocate C-array and set new length
            kmer_ptr = <uint32*>realloc(
                kmer_ptr,
                length * sizeof(uint32)
            )
            if not kmer_ptr:
                raise MemoryError
            (<int64*> kmer_ptr)[0] = length

            # Add k-mer location to C-array
            kmer_ptr[length-2] = reference_index
            kmer_ptr[length-1] = seq_pos

            # Store new pointer in pointer array
            ptr_array[kmer] = <ptr>kmer_ptr
    

    def add_kmers(self, kmer_sequence, reference_index, position):
        raise NotImplementedError()
        # TODO
    

    def get_kmers(self):
        raise NotImplementedError()
        # TODO
    

    def merge(self, kmer_table):
        raise NotImplementedError()


    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    def match(self, KmerTable kmer_table):
        """
        Notes
        -----

        There is no guaranteed order of the sequence indices or
        sequence positions in the returned matches.
        """
        cdef int INIT_SIZE = 1
        
        cdef int64 kmer
        cdef int64 match_i
        cdef int64 i, j
        cdef int64 self_length, other_length
        cdef uint32* self_kmer_ptr
        cdef uint32* other_kmer_ptr

        # Store in new variables
        # to disable repetitive initialization checks
        cdef ptr[:] self_ptr_array = self._ptr_array
        cdef ptr[:] other_ptr_array = kmer_table._ptr_array

        if self._kmer_alph != kmer_table._kmer_alph:
            raise ValueError(
                "The alphabet used for this k-mer index table is not equal to "
                "the alphabet of the other k-mer index table"
            )
        if self._k != kmer_table._k:
            raise ValueError(
                "The 'k' parameter used for this k-mer index table is not "
                "equal to 'k' of the other k-mer index"
            )

        cdef uint32[:,:] matches = np.zeros((INIT_SIZE, 4), dtype=np.uint32)
        match_i = 0
        for kmer in range(self_ptr_array.shape[0]):
            self_kmer_ptr = <uint32*>self_ptr_array[kmer]
            other_kmer_ptr = <uint32*>other_ptr_array[kmer]
            # For each k-mer create the cartesian product
            if self_kmer_ptr != NULL and other_kmer_ptr != NULL:
                # This kmer exists for both tables
                other_length = (<int64*>other_kmer_ptr)[0]
                self_length  = (<int64*>self_kmer_ptr )[0]
                for i in range(2, other_length, 2):
                    for j in range(2, self_length, 2):
                        if match_i >= matches.shape[0]:
                            matches = increase_array_size(np.asarray(matches))
                        matches[match_i, 0] = self_kmer_ptr[i]
                        matches[match_i, 1] = other_kmer_ptr[j]
                        matches[match_i, 2] = self_kmer_ptr[i+1]
                        matches[match_i, 3] = other_kmer_ptr[j+1]
                        match_i += 1

        return np.asarray(matches[:match_i])
    

    def update(self, kmer, seq_indices, positions):
        raise NotImplementedError()
        # TODO
    

    def __setitem__(self, kmer, item):
        raise NotImplementedError()
        # TODO


    def __getitem__(self, kmer):
        raise NotImplementedError()
        # TODO
    

    def __str__(self):
        cdef int64 kmer
        cdef int64 i
        cdef int64 length
        
        lines = []
        for kmer in range(self._ptr_array.shape[0]):
            if self._ptr_array[kmer] != 0:
                position_strings = []
                length = (<int64*>self._ptr_array[kmer])[0]
                for i in range(2, length, 2):
                    position_strings.append(str((
                        (<uint32*>self._ptr_array[kmer])[i],
                        (<uint32*>self._ptr_array[kmer])[i+1],
                    )))
                symbols = self._kmer_alph.decode(kmer)
                if isinstance(self._kmer_alph.base_alphabet, LetterAlphabet):
                    symbols = "".join(symbols)
                else:
                    symbols = str(tuple(symbols))
                line = symbols + ": " + ", ".join(position_strings)
                lines.append(line)
        return "\n".join(lines)


    def __dealloc__(self):
        if self._is_initialized():
            _deallocate_ptrs(self._ptr_array)
    
    cdef inline bint _is_initialized(self):
        # Memoryviews are not initialized on class creation
        # This method checks if the _kmer_len memoryview was initialized
        # and is not None
        try:
            if self._kmer_len is not None:
                return True
            else:
                return False
        except AttributeError:
            return False

    
cdef inline void _deallocate_ptrs(ptr[:] ptrs):
    cdef int i
    for i in range(ptrs.shape[0]):
        free(<int*>ptrs[i])


cdef np.ndarray increase_array_size(np.ndarray array):
    """
    Double the size of the first dimension of an existing array.
    """
    new_array = np.empty((array.shape[0]*2, array.shape[1]), dtype=array.dtype)
    new_array[:array.shape[0],:] = array
    return new_array