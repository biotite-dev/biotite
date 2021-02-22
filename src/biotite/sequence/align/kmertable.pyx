# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["KmerTable"]

cimport cython
cimport numpy as np
from libc.stdlib cimport realloc, malloc, free
from libc.string cimport memcpy

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
    

    def copy(self):
        clone = KmerTable(self._kmer_alph.base_alphabet, self._k)
        clone.merge(self)
        return clone

    
    @cython.boundscheck(False)
    @cython.wraparound(False)
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
        if not self._kmer_alph.base_alphabet.extends(sequence.alphabet):
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
    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def merge(self, KmerTable table):
        """
        If this function raises a :class:`MemoryError` this
        :class:`KmerTable` becomes invalid.
        """
        cdef int64 kmer
        cdef int64 self_length, other_length, new_length
        cdef uint32* self_kmer_ptr
        cdef uint32* other_kmer_ptr
        cdef int64 i, j

        # Store in new variables
        # to disable repetitive initialization checks
        cdef ptr[:] self_ptr_array = self._ptr_array
        cdef ptr[:] other_ptr_array = table._ptr_array

        if self._kmer_alph.base_alphabet != table._kmer_alph.base_alphabet:
            raise ValueError(
                "The alphabet used for this k-mer index table is not equal to "
                "the alphabet of the other k-mer index table"
            )
        if self._k != table._k:
            raise ValueError(
                "The 'k' parameter used for this k-mer index table is not "
                "equal to 'k' of the other k-mer index"
            )
        
        for kmer in range(self_ptr_array.shape[0]):
            self_kmer_ptr = <uint32*>self_ptr_array[kmer]
            other_kmer_ptr = <uint32*>other_ptr_array[kmer]
            if other_kmer_ptr != NULL:
                other_length = (<int64*>other_kmer_ptr)[0]
                if self_kmer_ptr == NULL:
                    # No entry yet, but the space for the array length
                    # value is still required
                    self_length  = 2
                else:
                    self_length  = (<int64*>self_kmer_ptr)[0]
                # New new C-array needs the combined space of both
                # arrays, but only one length value
                new_length = self_length + other_length - 2
                
                # Reallocate C-array and set new length
                self_kmer_ptr = <uint32*>realloc(
                    self_kmer_ptr,
                    new_length * sizeof(uint32)
                )
                if not self_kmer_ptr:
                    raise MemoryError
                (<int64*> self_kmer_ptr)[0] = new_length

                # Copy all entries from the other table into this table
                i = self_length
                for j in range(2, other_length):
                    self_kmer_ptr[i] = other_kmer_ptr[j]
                    i += 1
                
                # Store new pointer in pointer array
                self_ptr_array[kmer] = <ptr>self_kmer_ptr
            


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def match(self, KmerTable table, similarity_rule=None):
        """
        For each equal (or similar, if `similarity_rule` is given,)
        k-mer the cartesian product of the entries is calculated.

        Notes
        -----

        There is no guaranteed order of the sequence indices or
        sequence positions in the returned matches.
        """
        cdef int INIT_SIZE = 1
        
        cdef int64 kmer, sim_kmer
        cdef int64 match_i
        cdef int64 i, j, l
        cdef int64 self_length, other_length
        cdef uint32* self_kmer_ptr
        cdef uint32* other_kmer_ptr

        # This variable will only be used if a similarity rule exists
        cdef int64[:] similar_kmers

        # Store in new variables
        # to disable repetitive initialization checks
        cdef ptr[:] self_ptr_array = self._ptr_array
        cdef ptr[:] other_ptr_array = table._ptr_array

        if self._kmer_alph.base_alphabet != table._kmer_alph.base_alphabet:
            raise ValueError(
                "The alphabet used for this k-mer index table is not equal to "
                "the alphabet of the other k-mer index table"
            )
        if self._k != table._k:
            raise ValueError(
                "The 'k' parameter used for this k-mer index table is not "
                "equal to 'k' of the other k-mer index"
            )

        # This array will store the match positions
        # As the final number of matches is unknown, a list-like
        # approach is used:
        # The array is initialized with a relatively small inital size
        # and every time the limit would be exceeded its size is doubled
        cdef uint32[:,:] matches = np.zeros((INIT_SIZE, 4), dtype=np.uint32)
        match_i = 0
        if similarity_rule is None:
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
                                # The 'matches' array is full
                                # -> double its size
                                matches = expand(np.asarray(matches))
                            matches[match_i, 0] = other_kmer_ptr[i]
                            matches[match_i, 1] = self_kmer_ptr[j]
                            matches[match_i, 2] = other_kmer_ptr[i+1]
                            matches[match_i, 3] = self_kmer_ptr[j+1]
                            match_i += 1
        else:
            for kmer in range(self_ptr_array.shape[0]):
                other_kmer_ptr = <uint32*>other_ptr_array[kmer]
                if other_kmer_ptr != NULL:
                    # If a similarity rule exists, iterate not only over
                    # the exact k-mer, but over all k-mers similar to
                    # the current k-mer
                    similar_kmers = similarity_rule.similar_kmers(
                        self._kmer_alph, kmer
                    )
                    for l in range(similar_kmers.shape[0]):
                        sim_kmer = similar_kmers[l]
                        # Actual copy of the code from the other
                        # if-Branch:
                        # It cannot be put properly in a cdef-function,
                        # as every function call would perform reference
                        # count changes and would decrease performance 
                        self_kmer_ptr = <uint32*>self_ptr_array[sim_kmer]
                        if self_kmer_ptr != NULL:
                            other_length = (<int64*>other_kmer_ptr)[0]
                            self_length  = (<int64*>self_kmer_ptr )[0]
                            for i in range(2, other_length, 2):
                                for j in range(2, self_length, 2):
                                    if match_i >= matches.shape[0]:
                                        matches = expand(np.asarray(matches))
                                    matches[match_i, 0] = self_kmer_ptr[j]
                                    matches[match_i, 1] = other_kmer_ptr[i]
                                    matches[match_i, 2] = self_kmer_ptr[j+1]
                                    matches[match_i, 3] = other_kmer_ptr[i+1]
                                    match_i += 1

        # Trim to correct size and return
        return np.asarray(matches[:match_i])
    

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    def match_sequence(self, sequence, similarity_rule=None, mask=None):
        raise NotImplementedError


    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    def match_kmers(self, kmers, similarity_rule=None, mask=None):
        raise NotImplementedError
    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_kmers(self):
        cdef int64 kmer
        cdef ptr[:] ptr_array = self._ptr_array
        
        # Pessimistic allocation:
        # The maximum number of used kmers are all possible kmers
        cdef int64[:] kmers = np.zeros(ptr_array.shape[0], dtype=np.int64)

        cdef int64 i = 0
        for kmer in range(ptr_array.shape[0]):
            if <uint32*>ptr_array[kmer] != NULL:
                kmers[i] = kmer
                i += 1
        
        # Trim to correct size
        return np.asarray(kmers)[:i]

    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update(self, int64 kmer, positions):
        cdef int64 length
        cdef uint32* kmer_ptr
        cdef int64 i, j

        if kmer >= len(self):
            raise IndexError(
                f"k-mer {kmer} is out of bounds for this table "
                f"containing {len(self)} k-mers"
            )

        cdef uint32[:,:] pos = np.asarray(positions, dtype=np.uint32)

        # Store in new variable
        # to disable repetitive initialization checks
        cdef ptr[:] ptr_array = self._ptr_array
        
        kmer_ptr = <uint32*>ptr_array[kmer]
        if kmer_ptr == NULL:
            # No entry yet, but the space for the array length
            # value is still required
            length = 2
        else:
            length = (<int64*>kmer_ptr)[0]
        new_length = length + pos.shape[0] * 2
        
        # Reallocate C-array and set new length
        kmer_ptr = <uint32*>realloc(
            kmer_ptr,
            new_length * sizeof(uint32)
        )
        if not kmer_ptr:
            raise MemoryError
        (<int64*> kmer_ptr)[0] = new_length

        # Copy all entries from the given ndarray into this C-array
        i = length
        for j in range(pos.shape[0]):
            kmer_ptr[i]   = pos[j, 0]
            kmer_ptr[i+1] = pos[j, 1]
            i += 2
        
        # Store new pointer in pointer array
        ptr_array[kmer] = <ptr>kmer_ptr
    

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __getitem__(self, int64 kmer):
        cdef int64 i, j
        cdef int64 length
        cdef uint32* kmer_ptr
        cdef uint32[:,:] positions
        
        if kmer >= len(self):
            raise IndexError(
                f"k-mer {kmer} is out of bounds for this table "
                f"containing {len(self)} k-mers"
            )
        
        kmer_ptr = <uint32*>self._ptr_array[kmer]
        if kmer_ptr == NULL:
            return np.zeros((0, 2), dtype=np.uint32)
        else:
            length = (<int64*>kmer_ptr)[0]
            positions = np.empty(((length - 2) // 2, 2), dtype=np.uint32)
            i = 0
            for j in range(2, length, 2):
                positions[i,0] = kmer_ptr[j]
                positions[i,1] = kmer_ptr[j+1]
                i += 1
            return np.asarray(positions)


    def __setitem__(self, int64 kmer, positions):
        del self[kmer]
        self.update(kmer, positions)
    

    def __delitem__(self, int64 kmer):
        if kmer >= len(self):
            raise IndexError(
                f"k-mer {kmer} is out of bounds for this table "
                f"containing {len(self)} k-mers"
            )
        
        free(<uint32*>self._ptr_array[kmer])
        self._ptr_array[kmer] = 0


    def __contains__(self, kmer):
        return (kmer >= 0) & (kmer < len(self))

    
    def __len__(self):
        return len(self._kmer_alph)
    

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


    def __reversed__(self):
        for i in reversed(range(len(self))):
            yield self[i]
       

    def __eq__(self, item):
        if item is self:
            return True
        if type(item) != KmerTable:
            return False
        return self._equals(item)
    
    def _equals(self, KmerTable other):
        cdef int64 kmer
        cdef int64 i
        cdef int64 self_length, other_length
        cdef uint32* self_kmer_ptr
        cdef uint32* other_kmer_ptr

        # Store in new variables
        # to disable repetitive initialization checks
        cdef ptr[:] self_ptr_array = self._ptr_array
        cdef ptr[:] other_ptr_array = other._ptr_array

        if self._kmer_alph.base_alphabet != other._kmer_alph.base_alphabet:
            return False
        if self._k != other._k:
            return False

        for kmer in range(self_ptr_array.shape[0]):
            self_kmer_ptr = <uint32*>self_ptr_array[kmer]
            other_kmer_ptr = <uint32*>other_ptr_array[kmer]
            if self_kmer_ptr != NULL or other_kmer_ptr != NULL:
                if self_kmer_ptr == NULL or other_kmer_ptr == NULL:
                    # One of the tables has entries for this k-mer
                    # while the other has not
                    return False
                # This kmer exists for both tables
                self_length  = (<int64*>self_kmer_ptr )[0]
                other_length = (<int64*>other_kmer_ptr)[0]
                if self_length != other_length:
                    return False
                for i in range(2, self_length):
                    if self_kmer_ptr[i] != other_kmer_ptr[i]:
                        return False
        
        # If none of the previous checks failed, both objects are equal
        return True
    

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
    

    def __getnewargs_ex__(self):
        return (self._kmer_alph.base_alphabet, self._k), {}
    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __getstate__(self):
        """
        Pointer arrays.
        """
        cdef int64 i
        cdef list pickled_pointers = [b""] * len(self)
        cdef ptr[:] ptr_array = self._ptr_array
        cdef uint32* kmer_ptr

        for kmer in range(ptr_array.shape[0]):
            kmer_ptr = <uint32*>ptr_array[kmer]
            length = (<int64*>kmer_ptr)[0]
            if kmer_ptr != NULL:
                pickled_pointers[kmer] \
                    = <bytes>(<char*>kmer_ptr)[:sizeof(uint32) * length]
        
        return pickled_pointers
    

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __setstate__(self, state):
        print(len(self._ptr_array))
        cdef int64 i
        cdef int64 byte_length
        cdef uint32* kmer_ptr
        cdef bytes pickled_bytes

        cdef list pickled_pointers = state
        if len(pickled_pointers) != len(self._ptr_array):
            raise ValueError("Invalid pointer array found while unpickling")
        
        cdef ptr[:] ptr_array = self._ptr_array
        for kmer in range(ptr_array.shape[0]):
            pickled_bytes = pickled_pointers[kmer]
            byte_length = len(pickled_bytes)
            if byte_length != 0:
                kmer_ptr = <uint32*>malloc(byte_length)
                if not kmer_ptr:
                    raise MemoryError
                # Convert byte length to length of uint32 values
                (<int64*> kmer_ptr)[0] = byte_length // 4
                memcpy(kmer_ptr, <char*>pickled_bytes, byte_length)
                ptr_array[kmer] = <ptr>kmer_ptr


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
    cdef int kmer
    for kmer in range(ptrs.shape[0]):
        free(<uint32*>ptrs[kmer])


cdef np.ndarray expand(np.ndarray array):
    """
    Double the size of the first dimension of an existing array.
    """
    new_array = np.empty((array.shape[0]*2, array.shape[1]), dtype=array.dtype)
    new_array[:array.shape[0],:] = array
    return new_array