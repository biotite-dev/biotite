# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["KmerIndex", "kmer_number"]

cimport cython
cimport numpy as np
from libc.stdlib cimport realloc, malloc, free

import numpy as np
from ..alphabet import AlphabetError


ctypedef np.int32_t int32
ctypedef np.int64_t int64
ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64
ctypedef np.uint64_t ptr

ctypedef fused CodeType:
    uint8
    #uint16
    #uint32
    #uint64


cdef class KmerIndex:

    cdef object _alph
    cdef int64[:,:] _sim_kmers
    cdef bint _has_sim_rule
    cdef int _k
    cdef int _n_sequences

    cdef int64[:] _kmer_len
    cdef ptr[:] _kmer_seq
    cdef ptr[:] _kmer_pos


    def __cinit__(self, alphabet, k, similarity_rule=None):
        # This check is necessary for proper memory management
        # of the allocated arrays
        if self._is_initialized():
            raise Exception("Duplicate call of constructor")
        
        if k < 2:
            raise ValueError("k must be at least 2")
        if len(alphabet) < 1:
            raise ValueError("Alphabet length must be at least 1")
        
        self._alph = alphabet
        n_kmers = kmer_number(k, len(alphabet))

        if similarity_rule is None:
            self._has_sim_rule = False
        else:
            similar_kmers = similarity_rule.similarities(k, alphabet)
            if similar_kmers.ndim != 2:
                raise ValueError(
                    f"Got an {similar_kmers.ndim}-dimensional array for "
                    f"for similar kmers, expected 2 dimensions"
                )
            if len(similar_kmers) > n_kmers:
                raise ValueError(
                    f"k-mer similarity array has {len(similar_kmers)}, "
                    f"but there are {n_kmers}, must be equal"
                )
            max_kmer = np.max(similar_kmers)
            if max_kmer >= n_kmers:
                raise ValueError(
                    f"k-mer similarity array maps to kmer "
                    f"{len(similar_kmers)}, but there are only {n_kmers}"
                )
            self._sim_kmers = similar_kmers.astype(np.int64)
            self._has_sim_rule = True
        
        self._k = k
        self._n_sequences = 0

        # ndarrays of pointers to C-arrays
        self._kmer_seq = np.zeros(n_kmers, dtype=np.uint64)
        self._kmer_pos = np.zeros(n_kmers, dtype=np.uint64)
        # Stores the length of the C-arrays
        self._kmer_len = np.zeros(n_kmers, dtype=np.int64)


    @property
    def alphabet(self):
        return self._alphabet
    
    @property
    def k(self):
        return self._k

    @property
    def sequence_number(self):
        return self._n_sequences

    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add(self, sequence, mask=None):
        """
        If this function raises a :class:`MemoryError` the
        :class:`KmerIndex` becomes invalid.
        """
        cdef int i
        cdef int64 seq_pos
        cdef int64 length
        cdef int64 kmer, sim_kmer
        cdef int64* kmer_ptr

        # Store in new variables
        # to disable repetitive initialization checks
        cdef int64[:,:] sim_kmers
        cdef int64[:] kmer_len = self._kmer_len
        cdef ptr[:] kmer_seq = self._kmer_seq
        cdef ptr[:] kmer_pos = self._kmer_pos
        cdef int n_sequences = self._n_sequences

        code = sequence.code
        
        if len(code) < self._k:
            raise ValueError("Sequence code is shorter than k")
        if self._alph != sequence.alphabet:
            raise ValueError(
                "The alphabet used for the k-mer index is not equal to the "
                "alphabet of the sequence"
            )
        
        cdef int64[:] kmers = _create_kmers(
            code, self._k, len(self._alph)
        )
        
        if self._has_sim_rule:
            sim_kmers = self._sim_kmers
            for seq_pos in range(kmers.shape[0]):
                kmer = kmers[seq_pos]
                # There is a similarity rule
                # -> dd not only this k-mer to the index,
                # but also all similar kmers
                # As the array of similar k-mers should include
                # self-similarity, this k-mer should also be included
                # in the array
                for i in range(sim_kmers.shape[1]):
                    sim_kmer = sim_kmers[kmer, i]
                    if sim_kmer < 0:
                        # Reached the trailing '-1' values
                        break
                    # Add the k-mer to the index
                    # and watch out for MemoryError in doing so
                    if _add_kmer(
                        kmer, n_sequences, seq_pos,
                        kmer_len, kmer_seq, kmer_pos
                    ):
                        raise MemoryError
        else:
            for seq_pos in range(kmers.shape[0]):
                kmer = kmers[seq_pos]
                # No similarity rule
                # -> do not add similar k-mers to the index
                # Add the k-mer to the index
                # and watch out for MemoryError in doing so
                if _add_kmer(
                    kmer, n_sequences, seq_pos,
                    kmer_len, kmer_seq, kmer_pos
                ):
                    raise MemoryError
            

        
        self._n_sequences += 1
    

    

    def add_kmers(self, kmer_sequence, seq_index, position):
        raise NotImplementedError()
        # TODO
    

    def get_kmers(self):
        raise NotImplementedError()
        # TODO
    

    def merge(self, kmer_index):
        raise NotImplementedError()


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def match(self, KmerIndex kmer_index):
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
        cdef int64* self_seq_ptr
        cdef int64* self_pos_ptr
        cdef int64* other_seq_ptr
        cdef int64* other_pos_ptr

        # Store in new variables
        # to disable repetitive initialization checks
        cdef int64[:] self_kmer_len = self._kmer_len
        cdef ptr[:] self_kmer_seq = self._kmer_seq
        cdef ptr[:] self_kmer_pos = self._kmer_pos
        cdef int64[:] other_kmer_len = kmer_index._kmer_len
        cdef ptr[:] other_kmer_seq = kmer_index._kmer_seq
        cdef ptr[:] other_kmer_pos = kmer_index._kmer_pos

        if self._alph != kmer_index._alph:
            raise ValueError(
                "The alphabet used for this k-mer index is not equal to the "
                "alphabet of the other k-mer index"
            )

        cdef int64[:,:] matches = np.zeros((INIT_SIZE, 4), dtype=np.int64)
        match_i = 0
        for kmer in range(self_kmer_len.shape[0]):
            self_seq_ptr = <int64*>self_kmer_seq[kmer]
            self_pos_ptr = <int64*>self_kmer_pos[kmer]
            other_seq_ptr = <int64*>other_kmer_seq[kmer]
            other_pos_ptr = <int64*>other_kmer_pos[kmer]
            # For each k-mer create the cartesian product
            if self_kmer_len[kmer] != 0 or other_kmer_len[kmer] != 0:
                # This kmer exists for both indexes
                for i in range(self_kmer_len[kmer]):
                    for j in range(other_kmer_len[kmer]):
                        if match_i >= matches.shape[0]:
                            matches = increase_array_size(np.asarray(matches))
                        matches[match_i, 0] = self_seq_ptr[i]
                        matches[match_i, 1] = other_seq_ptr[j]
                        matches[match_i, 2] = self_pos_ptr[i]
                        matches[match_i, 3] = other_pos_ptr[j]
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
        
        lines = []
        for kmer in range(self._kmer_len.shape[0]):
            if self._kmer_len[kmer] > 0:
                position_strings = []
                for i in range(self._kmer_len[kmer]):
                    position_strings.append(str((
                        (<int64*>self._kmer_seq[kmer])[i],
                        (<int64*>self._kmer_pos[kmer])[i],
                    )))
                symbols = self._alph.decode_multiple(self._to_seq_code(kmer))
                line = str(tuple(symbols)) + ": " + ", ".join(position_strings)
                lines.append(line)
        return "\n".join(lines)


    def __dealloc__(self):
        if self._is_initialized():
            _deallocate_ptrs(self._kmer_seq)
            _deallocate_ptrs(self._kmer_pos)
    
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
    

    def _to_kmer(self, seq_code):
        radix_multiplier = np.array(
            [len(self._alph)**n for n in range(self._k)],
            dtype=int
        )

        if len(seq_code) != self._k:
            raise ValueError(
                f"Sequence code has length {len(seq_code)}, "
                f"but k-mer index expects {self._k}"
            )
        
        return np.sum(self._radix_multiplier * seq_code)
    
    cdef _to_seq_code(self, kmer):
        radix_multiplier = np.array(
            [len(self._alph)**n for n in range(self._k)],
            dtype=int
        )

        if kmer >= kmer_number(self._k, len(self._alph)) or kmer < 0:
            raise ValueError(
                f"k-mer {kmer} is invalid for the underlying alphabet"
            )
        seq_code = np.zeros(self._k, dtype=int)
        for n in reversed(range(self._k)):
            val = radix_multiplier[n]
            code = kmer // val
            seq_code[n] = code
            kmer -= code * val
        return seq_code


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _add_kmer(int64 kmer, int64 seq_index, int64 seq_pos,
                           int64[:] kmer_len,
                           ptr[:] kmer_seq, ptr[:] kmer_pos):
    cdef int64 length = kmer_len[kmer] + 1
    kmer_len[kmer] = length

    # Increment array length for this kmer and reallocate
    # Set the index of the sequence in the list for this kmer
    cdef int64* kmer_ptr = <int64*>kmer_seq[kmer]
    kmer_ptr = <int64*>realloc(kmer_ptr, length * sizeof(int64))
    if not kmer_ptr:
        # '1' is indicator for MemoryError
        # No exception is raised directly
        # to keep efficiency of inline function call 
        return 1
    kmer_ptr[length-1] = seq_index
    kmer_seq[kmer] = <ptr>kmer_ptr

    # Set the position of this kmer in the sequence
    kmer_ptr = <int64*>kmer_pos[kmer]
    kmer_ptr = <int64*>realloc(kmer_ptr, length * sizeof(int64))
    if not kmer_ptr:
        # '1' is indicator for MemoryError
        # No exception is raised directly
        # to keep efficiency of inline function call 
        return 1
    kmer_ptr[length-1] = seq_pos
    kmer_pos[kmer] = <ptr>kmer_ptr
    

    
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


@cython.boundscheck(False)
@cython.wraparound(False)
def _create_kmers(CodeType[:] seq_code not None,
                  int k, int64 alphabet_length):
    cdef int64 i
    cdef int j
    
    cdef int64[:] radix_multiplier = np.array(
        [alphabet_length**n for n in range(k)],
        dtype=np.int64
    )

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


def kmer_number(k, alphabet_length):
    return alphabet_length**k