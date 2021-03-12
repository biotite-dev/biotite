# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["KmerTable"]

cimport cython
cimport numpy as np
from cpython.mem cimport PyMem_Malloc as malloc
from cpython.mem cimport PyMem_Realloc as realloc
from cpython.mem cimport PyMem_Free as free
from libc.string cimport memcpy

import numpy as np
from ..alphabet import LetterAlphabet
from .kmeralphabet import KmerAlphabet


ctypedef np.int32_t int32
ctypedef np.int64_t int64
ctypedef np.uint8_t uint8
ctypedef np.uint32_t uint32
ctypedef np.uint64_t ptr


cdef class KmerTable:
    """
    This class represents a *k-mer* index table.
    It maps *k-mers* (subsequences with length *k*) to the sequence
    positions, where the *k-mer* appears.
    It is primarily used to find *k-mer* matches between two sequences.
    A match is defined as a *k-mer* that appears in both sequences.

    A sequence can be indexed via the :meth:`add()` method, which takes
    a :class:`Sequence` and a reference ID.
    The reference ID is a unique integer, that identifies to which
    sequence a match position refers to, in case multiple sequences
    are added to the :class:`KmerTable`.
    This method iterates through all overlapping *k-mers* in the
    sequence and, for each *k-mer*, appends the zero-based sequence
    position of the first symbol in the *k-mer* along with the
    reference ID to the table entry of the *k-mer*.

    After adding at least one sequence, the :meth:`match()` method
    iterates through all overlapping *k-mers* in another sequence and,
    for each *k-mer*, looks up the reference IDs and positions of this
    *k-mer* in the previously added sequences.
    For each matching position, it adds the *k-mer* position in this
    sequence, the matching reference ID and the matching sequence
    position to the array of matches.
    Finally these matches are returned to the user.

    :class:`KmerTable` objects can be handled similar to a fixed-sized
    list for accessing and editing purposes:
    Indexing a table with a *k-mer* code returns an array
    of all reference IDs and positions, where this *k-mer* appears.
    Conversely, setting the table at a given *k-mer* code overwrites
    the reference IDs and positions for this *k-mer* with new ones.
    The :meth:`update()` method manually appends new reference IDs and
    positions to the existing ones, instead of overwriting them.
    The *k-mer* code for a *k-mer* can be calculated with
    ``table.kmer_alphabet.encode()`` (see :class:`KmerAlphabet`).

    Parameters
    ----------
    alphabet : Alphabet
        The base alphabet that defines the kind of sequences that are
        added to / matched by this table.
        It must be equal to or extend the alphabet of all
        sequences that are added to the table or are matched.
    k : int
        An integer greater than 1 that defines the *k-mer* length.
    
    Attributes
    ----------
    kmer_alphabet : KmerAlphabet
        The internal :class:`KmerAlphabet`, that is used to
        encode all overlapping *k-mers* of an input sequence.
    alphabet : Alphabet
        The base alphabet, from which this :class:`KmerTable` was
        created.
    k : int
        The length of the *k-mers*.

    Notes
    -----

    **Memory consumption**

    For efficient mapping, a :class:`KmerTable` contains a pointer
    array, that contains one 64-bit pointer for each possible *k-mer*.
    If there is at least one match for a *k-mer*, the corresponding
    pointer points to a C-array that contains

        1. The length of the C-array *(int64)*
        2. The reference ID for each match of this *k-mer* *(uint32)*
        3. The sequence position for each match of this *k-mer* *(uint32)*
    
    Hence, the memory requirements can be quite large for long *k-mers*
    or large alphabets.
    The required memory space :math:`S` in byte is within the bounds of

    .. math::

        8 n^k + 8L \leq S \leq 16 n^k + 8L,

    where :math:`n` is the number of symbols in the alphabet and
    :math:`L` is the summed length of all sequences added to the table.

    **Multiprocessing**

    :class:`KmerTable` objects can be used in multi-processed setups:
    Adding a large database of sequences to a table can be sped up by
    splitting the database into smaller chunks and adding each chunk
    to a separate table in separate processes.
    Eventually, the tables can be merged to one large table using
    :meth:`merge()`.

    Since :class:`KmerTable` supports the *pickle* protocol,
    the matching step can also be divided into multiple processes, if
    multiple sequences need to be matched.

    **Storage on hard drive**

    The most time efficient way to read/write a :class:`KmerTable` is
    the *pickle* format.
    If a custom format is desired, the user needs to extract the
    reference IDs and position for each *k-mer*. To restrict this task
    to all *k-mer* that have at least one match :meth:`get_kmers()` can
    be used.
    Conversely, the reference IDs and position can be restored via
    :meth:`update()`, which is slightly faster than indexing.

    Examples
    --------

    Create a *k-mer* index table for unambiguous nucleotide sequences:

    >>> table = KmerTable(NucleotideSequence.unambiguous_alphabet(), k=2)

    Add some sequences to the table:

    >>> table.add(NucleotideSequence("TTATA"), ref_id=0)
    >>> table.add(NucleotideSequence("CTAG"),  ref_id=1)

    Display the contents of the table as
    (reference ID, sequence position) tuples:

    >>> print(table)
    TA: (0, 1), (0, 3), (1, 1)
    AG: (1, 2)
    AT: (0, 2)
    CT: (1, 0)
    TT: (0, 0)

    Find matches of the table with a sequence:

    >>> query = NucleotideSequence("TAG")
    >>> matches = table.match(query)
    >>> for query_pos, table_ref_id, table_pos in matches:
    ...     print("Query sequence position:", query_pos)
    ...     print("Table reference ID:  ", table_ref_id)
    ...     print("Table sequence position:", table_pos)
    ...     print()
    Query sequence position: 0
    Table reference ID: 0
    Table sequence position: 1
    <BLANKLINE>
    Query sequence position: 0
    Table reference ID: 0
    Table sequence position: 3
    <BLANKLINE>
    Query sequence position: 0
    Table reference ID: 1
    Table sequence position: 1
    <BLANKLINE>
    Query sequence position: 1
    Table reference ID: 1
    Table sequence position: 2
    <BLANKLINE>

    Get all reference IDs and positions for a given *k-mer*:

    >>> kmer_code = table.kmer_alphabet.encode("TA")
    >>> print(table[kmer_code])
    [[0 1]
     [0 3]
     [1 1]]

    Edit the table manually:

    >>> table.update(table.kmer_alphabet.encode("TA"), [[2, 42]])
    >>> table[table.kmer_alphabet.encode("CT")] = [[3, 100], [4, 200]]
    >>> print(table)
    TA: (0, 1), (0, 3), (1, 1), (2, 42)
    AG: (1, 2)
    AT: (0, 2)
    CT: (3, 100), (4, 200)
    TT: (0, 0)
    """

    cdef object _kmer_alph
    cdef int _k

    # The pointer array is the core of the index table:
    # It maps each possible k-mer (represented by its code) to a
    # C-array of indices.
    # Each entry in a C-array points to a reference ID and the
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
    def add(self, sequence, uint32 ref_id, removal_mask=None):
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
        
        if len(sequence.code) < self._k:
            raise ValueError("Sequence code is shorter than k")
        if not self._kmer_alph.base_alphabet.extends(sequence.alphabet):
            raise ValueError(
                "The alphabet used for the k-mer index table is not equal to "
                "the alphabet of the sequence"
            )
        
        cdef int64[:] kmers = self._kmer_alph.create_kmers(sequence.code)
        cdef uint8[:] kmer_mask = self._prepare_mask(
            removal_mask, len(sequence.code)
        )

        for seq_pos in range(kmers.shape[0]):
            if kmer_mask[seq_pos]:
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
                kmer_ptr[length-2] = ref_id
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
    def match_table(self, KmerTable table, similarity_rule=None):
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
        cdef int64[:,:] matches = np.empty((INIT_SIZE, 4), dtype=np.int64)
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
                            matches[match_i, 1] = other_kmer_ptr[i+1]
                            matches[match_i, 2] = self_kmer_ptr[j]
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
                                    matches[match_i, 0] = other_kmer_ptr[i]
                                    matches[match_i, 1] = other_kmer_ptr[i+1]
                                    matches[match_i, 2] = self_kmer_ptr[j]
                                    matches[match_i, 3] = self_kmer_ptr[j+1]
                                    match_i += 1

        # Trim to correct size and return
        return np.asarray(matches[:match_i])
    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def match(self, sequence, similarity_rule=None,
                       removal_mask=None):
        cdef int INIT_SIZE = 1
        
        cdef int64 kmer, sim_kmer
        cdef int64 match_i
        cdef int64 i, j, l
        cdef int64 length
        cdef uint32* kmer_ptr

        # This variable will only be used if a similarity rule exists
        cdef int64[:] similar_kmers

        # Store in new variable
        # to disable repetitive initialization checks
        cdef ptr[:] ptr_array = self._ptr_array

        if len(sequence.code) < self._k:
            raise ValueError("Sequence code is shorter than k")
        if not self._kmer_alph.base_alphabet.extends(sequence.alphabet):
            raise ValueError(
                "The alphabet used for the k-mer index table is not equal to "
                "the alphabet of the sequence"
            )
        
        cdef int64[:] kmers = self._kmer_alph.create_kmers(sequence.code)
        cdef uint8[:] kmer_mask = self._prepare_mask(
            removal_mask, len(sequence.code)
        )

        # This array will store the match positions
        # As the final number of matches is unknown, a list-like
        # approach is used:
        # The array is initialized with a relatively small inital size
        # and every time the limit would be exceeded its size is doubled
        cdef int64[:,:] matches = np.empty((INIT_SIZE, 3), dtype=np.int64)
        match_i = 0
        if similarity_rule is None:
            for i in range(kmers.shape[0]):
                if kmer_mask[i]:
                    kmer = kmers[i]
                    kmer_ptr = <uint32*>ptr_array[kmer]
                    if kmer_ptr != NULL:
                        # There is at least one entry for the k-mer
                        length = (<int64*>kmer_ptr)[0]
                        for j in range(2, length, 2):
                                if match_i >= matches.shape[0]:
                                    # The 'matches' array is full
                                    # -> double its size
                                    matches = expand(np.asarray(matches))
                                matches[match_i, 0] = i
                                matches[match_i, 1] = kmer_ptr[j]
                                matches[match_i, 2] = kmer_ptr[j+1]
                                match_i += 1
        
        else:
            for i in range(kmers.shape[0]):
                if kmer_mask[i]:
                    kmer = kmers[i]
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
                        kmer_ptr = <uint32*>ptr_array[sim_kmer]
                        if kmer_ptr != NULL:
                            # There is at least one entry for the k-mer
                            length = (<int64*>kmer_ptr)[0]
                            for j in range(2, length, 2):
                                    if match_i >= matches.shape[0]:
                                        # The 'matches' array is full
                                        # -> double its size
                                        matches = expand(np.asarray(matches))
                                    matches[match_i, 0] = i
                                    matches[match_i, 1] = kmer_ptr[j]
                                    matches[match_i, 2] = kmer_ptr[j+1]
                                    match_i += 1

        # Trim to correct size and return
        return np.asarray(matches[:match_i])


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
        
        if len(positions) == 0:
            # No position to add -> simply return and do not create an
            # unnecessary empty C-array
            return

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
        lines = []
        for kmer in self.get_kmers():
            symbols = self._kmer_alph.decode(kmer)
            if isinstance(self._kmer_alph.base_alphabet, LetterAlphabet):
                symbols = "".join(symbols)
            else:
                symbols = str(tuple(symbols))
            line = symbols + ": " + ", ".join(
                [str(tuple(pos)) for pos in self[kmer]]
            )
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
        cdef int64 kmer
        cdef uint32* kmer_ptr

        cdef ptr[:] ptr_array = self._ptr_array

        cdef int64[:] relevant_kmers = self.get_kmers()
        cdef list pickled_pointers = [b""] * relevant_kmers.shape[0]

        for i in range(relevant_kmers.shape[0]):
            kmer = relevant_kmers[i]
            kmer_ptr = <uint32*>ptr_array[kmer]
            length = (<int64*>kmer_ptr)[0]
            # Get directly the bytes coding for each C-array
            pickled_pointers[i] \
                = <bytes>(<char*>kmer_ptr)[:sizeof(uint32) * length]
        
        return np.asarray(relevant_kmers), pickled_pointers
    

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __setstate__(self, state):
        cdef int64 i
        cdef int64 kmer
        cdef int64 byte_length
        cdef uint32* kmer_ptr
        cdef bytes pickled_bytes

        cdef int64[:] relevant_kmers = state[0]
        cdef list pickled_pointers = state[1]
        
        cdef ptr[:] ptr_array = self._ptr_array
        for i in range(relevant_kmers.shape[0]):
            kmer = relevant_kmers[i]
            if kmer < 0 or kmer >= ptr_array.shape[0]:
                raise ValueError("Invalid k-mer found while unpickling")
            pickled_bytes = pickled_pointers[i]
            byte_length = len(pickled_bytes)
            if byte_length != 0:
                kmer_ptr = <uint32*>malloc(byte_length)
                if not kmer_ptr:
                    raise MemoryError
                # Convert bytes back into C-array
                memcpy(kmer_ptr, <char*>pickled_bytes, byte_length)
                ptr_array[kmer] = <ptr>kmer_ptr


    def __dealloc__(self):
        if self._is_initialized():
            _deallocate_ptrs(self._ptr_array)
    

    def _prepare_mask(self, removal_mask, seq_length):
        if removal_mask is None:
            kmer_mask = np.ones(seq_length - self._k + 1, dtype=np.uint8)
        else:
            if not isinstance(removal_mask, np.ndarray):
                raise TypeError(
                    f"The given mask is a '{type(removal_mask).__name__}', "
                    f"but an ndarray was expected"
                )
            if removal_mask.dtype != np.dtype(bool):
                raise ValueError("Expected a boolean mask")
            kmer_mask = _to_kmer_mask(
                np.frombuffer(removal_mask, dtype=np.uint8), self._k
            )
        return kmer_mask
    

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


#@cython.boundscheck(False)
#@cython.wraparound(False)
def _to_kmer_mask(uint8[:] mask not None, int64 k):
    cdef int64 i, j
    cdef bint is_retained
    
    cdef uint8[:] kmer_mask = np.empty(mask.shape[0] - k + 1, dtype=np.uint8)
    
    for i in range(kmer_mask.shape[0]):
        is_retained = True
        # If any sequence position of this k-mer is removed,
        # discard this k-mer position
        for j in range(i, i + k):
            if mask[j]:
                is_retained = False
        kmer_mask[i] = is_retained
    
    return np.asarray(kmer_mask)
    
