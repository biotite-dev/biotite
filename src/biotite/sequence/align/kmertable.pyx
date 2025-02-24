# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

# distutils: language = c++

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["KmerTable", "BucketKmerTable"]

cimport cython
cimport numpy as np
from cpython.mem cimport PyMem_Malloc as malloc
from cpython.mem cimport PyMem_Free as free
from libc.string cimport memcpy
from libcpp.set cimport set as cpp_set

import numpy as np
from ..alphabet import LetterAlphabet, common_alphabet, AlphabetError
from .kmeralphabet import KmerAlphabet
from .buckets import bucket_number


ctypedef np.int32_t int32
ctypedef np.int64_t int64
ctypedef np.uint8_t uint8
ctypedef np.uint32_t uint32
ctypedef np.uint64_t ptr


cdef enum EntrySize:
    # The size (number of 32 bit elements) for each entry in C-arrays
    # of KmerTable and BucketKmerTable, respectively
    #
    # Size: reference ID (int32) + sequence pos (int32)
    NO_BUCKETS = 2
    # Size: k-mer (int64) + reference ID (int32) + sequence pos (int32)
    BUCKETS = 4


cdef class KmerTable:
    """
    This class represents a *k-mer* index table.
    It maps *k-mers* (subsequences with length *k*) to the sequence
    positions, where the *k-mer* appears.
    It is primarily used to find *k-mer* matches between two sequences.
    A match is defined as a *k-mer* that appears in both sequences.
    Instances of this class are immutable.

    There are multiple ways to create a :class:`KmerTable`:

        - :meth:`from_sequences()` iterates through all overlapping
          *k-mers* in a sequence and stores the sequence position of
          each *kmer* in the table.
        - :meth:`from_kmers()` is similar to :meth:`from_sequences()`
          but directly accepts *k-mers* as input instead of sequences.
        - :meth:`from_kmer_selection()` takes a combination of *k-mers*
          and their positions in a sequence, which can be used to
          apply subset selectors, such as :class:`MinimizerSelector`.
        - :meth:`from_tables()` merges the entries from multiple
          :class:`KmerTable` objects into a new table.
        - :meth:`from_positions()` let's the user provide manual
          *k-mer* positions, which can be useful for loading a
          :class:`KmerTable` from file.

    The standard constructor merely returns an empty table and is
    reserved for internal use.

    Each indexed *k-mer* position is represented by a tuple of

        1. a unique reference ID that identifies to which sequence a
           position refers to and
        2. the zero-based sequence position of the first symbol in the
           *k-mer*.

    The :meth:`match()` method iterates through all overlapping *k-mers*
    in another sequence and, for each *k-mer*, looks up the reference
    IDs and positions of this *k-mer* in the table.
    For each matching position, it adds the *k-mer* position in this
    sequence, the matching reference ID and the matching sequence
    position to the array of matches.
    Finally these matches are returned to the user.
    Optionally, a :class:`SimilarityRule` can be supplied, to find
    also matches for similar *k-mers*.
    This is especially useful for protein sequences to match two
    *k-mers* with a high substitution probability.

    The positions for a given *k-mer* code can be obtained via indexing.
    Iteration over a :class:`KmerTable` yields the *k-mers* that have at
    least one associated position.
    The *k-mer* code for a *k-mer* can be calculated with
    ``table.kmer_alphabet.encode()`` (see :class:`KmerAlphabet`).

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

    See Also
    --------
    BucketKmerTable

    Notes
    -----

    The design of the :class:`KmerTable` is inspired by the *MMseqs2*
    software :footcite:`Steinegger2017`.

    *Memory consumption*

    For efficient mapping, a :class:`KmerTable` contains a pointer
    array, that contains one 64-bit pointer for each possible *k-mer*.
    If there is at least one position for a *k-mer*, the corresponding
    pointer points to a C-array that contains

        1. The length of the C-array *(int64)*
        2. The reference ID for each position of this *k-mer* *(uint32)*
        3. The sequence position for each position of this *k-mer* *(uint32)*

    Hence, the memory requirements can be quite large for long *k-mers*
    or large alphabets.
    The required memory space :math:`S` in byte is within the bounds of

    .. math::

        8 n^k + 8L \leq S \leq 16 n^k + 8L,

    where :math:`n` is the number of symbols in the alphabet and
    :math:`L` is the summed length of all sequences added to the table.

    *Multiprocessing*

    :class:`KmerTable` objects can be used in multi-processed setups:
    Adding a large database of sequences to a table can be sped up by
    splitting the database into smaller chunks and create a separate
    table for each chunk in separate processes.
    Eventually, the tables can be merged to one large table using
    :meth:`from_tables()`.

    Since :class:`KmerTable` supports the *pickle* protocol,
    the matching step can also be divided into multiple processes, if
    multiple sequences need to be matched.

    *Storage on hard drive*

    The most time efficient way to read/write a :class:`KmerTable` is
    the *pickle* format.
    If a custom format is desired, the user needs to extract the
    reference IDs and position for each *k-mer*.
    To restrict this task to all *k-mer* that have at least one match
    :meth:`get_kmers()` can be used.
    Conversely, the reference IDs and positions can be restored via
    :meth:`from_positions()`.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    Create a *2-mer* index table for some nucleotide sequences:

    >>> table = KmerTable.from_sequences(
    ...     k = 2,
    ...     sequences = [NucleotideSequence("TTATA"), NucleotideSequence("CTAG")],
    ...     ref_ids = [0, 1]
    ... )

    Display the contents of the table as
    (reference ID, sequence position) tuples:

    >>> print(table)
    AG: (1, 2)
    AT: (0, 2)
    CT: (1, 0)
    TA: (0, 1), (0, 3), (1, 1)
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
    # (Array length)  (RefID 0)   (Position 0)  (RefID 1)   (Position 1) ...
    # -----int64----|---uint32---|---uint32---|---uint32---|---uint32---
    #
    # The array length is based on 32 bit units.
    # If there is no entry for a k-mer, the respective pointer is NULL.
    cdef ptr[:] _ptr_array


    def __cinit__(self, kmer_alphabet):
        # This check is necessary for proper memory management
        # of the allocated arrays
        if self._is_initialized():
            raise Exception("Duplicate call of constructor")

        self._kmer_alph = kmer_alphabet
        self._k = kmer_alphabet.k
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

    @staticmethod
    def from_sequences(k, sequences, ref_ids=None, ignore_masks=None,
                       alphabet=None, spacing=None):
        """
        from_sequences(k, sequences, ref_ids=None, ignore_masks=None,
                       alphabet=None, spacing=None)

        Create a :class:`KmerTable` by storing the positions of all
        overlapping *k-mers* from the input `sequences`.

        Parameters
        ----------
        k : int
            The length of the *k-mers*.
        sequences : sized iterable object of Sequence, length=m
            The sequences to get the *k-mer* positions from.
            These sequences must have equal alphabets, or one of these
            sequences must have an alphabet that extends the alphabets
            of all other sequences.
        ref_ids : sized iterable object of int, length=m, optional
            The reference IDs for the given sequences.
            These are used to identify the corresponding sequence for a
            *k-mer* match.
            By default the IDs are counted from *0* to *m*.
        ignore_masks : sized iterable object of (ndarray, dtype=bool), length=m, optional
            Sequence positions to ignore.
            *k-mers* that involve these sequence positions are not added
            to the table.
            This is used e.g. to skip repeat regions.
            If provided, the list must contain one boolean mask
            (or ``None``) for each sequence, and each bolean mask must
            have the same length as the sequence.
            By default, no sequence position is ignored.
        alphabet : Alphabet, optional
            The alphabet to use for this table.
            It must extend the alphabets of the input `sequences`.
            By default, an appropriate alphabet is inferred from the
            input `sequences`.
            This option is usually used for compatibility with another
            sequence/table in the matching step.
        spacing : None or str or list or ndarray, dtype=int, shape=(k,)
            If provided, spaced *k-mers* are used instead of continuous
            ones.
            The value contains the *informative* positions relative to
            the start of the *k-mer*, also called the *model*.
            The number of *informative* positions must equal *k*.
            Refer to :class:`KmerAlphabet` for more details.

        See Also
        --------
        from_kmers : The same functionality based on already created *k-mers*

        Returns
        -------
        table : KmerTable
            The newly created table.

        Examples
        --------

        >>> sequences = [NucleotideSequence("TTATA"), NucleotideSequence("CTAG")]
        >>> table = KmerTable.from_sequences(
        ...     2, sequences, ref_ids=[100, 101]
        ... )
        >>> print(table)
        AG: (101, 2)
        AT: (100, 2)
        CT: (101, 0)
        TA: (100, 1), (100, 3), (101, 1)
        TT: (100, 0)

        Give an explicit compatible alphabet:

        >>> table = KmerTable.from_sequences(
        ...     2, sequences, ref_ids=[100, 101],
        ...     alphabet=NucleotideSequence.ambiguous_alphabet()
        ... )

        Ignore all ``N`` in a sequence:

        >>> sequence = NucleotideSequence("ACCNTANNG")
        >>> table = KmerTable.from_sequences(
        ...     2, [sequence], ignore_masks=[sequence.symbols == "N"]
        ... )
        >>> print(table)
        AC: (0, 0)
        CC: (0, 1)
        TA: (0, 4)
        """
        ref_ids = _compute_ref_ids(ref_ids, sequences)
        ignore_masks = _compute_masks(ignore_masks, sequences)
        alphabet = _compute_alphabet(
            alphabet, (sequence.alphabet for sequence in sequences)
        )

        table = KmerTable(KmerAlphabet(alphabet, k, spacing))

        # Calculate k-mers
        kmers_list = [
            table._kmer_alph.create_kmers(sequence.code)
            for sequence in sequences
        ]

        masks = [
            _prepare_mask(table._kmer_alph, ignore_mask, len(sequence))
            for sequence, ignore_mask in zip(sequences, ignore_masks)
        ]

        # Count the number of appearances of each k-mer and store the
        # result in the pointer array, that is now used as count array
        for kmers, mask in zip(kmers_list, masks):
            table._count_masked_kmers(kmers, mask)

        # Transfrom count array into pointer array with C-array of
        # appropriate size
        _init_c_arrays(table._ptr_array, EntrySize.NO_BUCKETS)

        # Fill the C-arrays with the k-mer positions
        for kmers, ref_id, mask in zip(kmers_list, ref_ids, masks):
            table._add_kmers(kmers, ref_id, mask)

        return table


    @staticmethod
    def from_kmers(kmer_alphabet, kmers, ref_ids=None, masks=None):
        """
        from_kmers(kmer_alphabet, kmers, ref_ids=None, masks=None)

        Create a :class:`KmerTable` by storing the positions of all
        input *k-mers*.

        Parameters
        ----------
        kmer_alphabet : KmerAlphabet
            The :class:`KmerAlphabet` to use for the new table.
            Should be the same alphabet that was used to calculate the
            input *kmers*.
        kmers : sized iterable object of (ndarray, dtype=np.int64), length=m
            List where each array contains the *k-mer* codes from a
            sequence.
            For each array the index of the *k-mer* code in the array
            is stored in the table as sequence position.
        ref_ids : sized iterable object of int, length=m, optional
            The reference IDs for the sequences.
            These are used to identify the corresponding sequence for a
            *k-mer* match.
            By default the IDs are counted from *0* to *m*.
        masks : sized iterable object of (ndarray, dtype=bool), length=m, optional
            A *k-mer* code at a position, where the corresponding mask
            is false, is not added to the table.
            By default, all positions are added.

        See Also
        --------
        from_sequences : The same functionality based on undecomposed sequences

        Returns
        -------
        table : KmerTable
            The newly created table.

        Examples
        --------

        >>> sequences = [ProteinSequence("BIQTITE"), ProteinSequence("NIQBITE")]
        >>> kmer_alphabet = KmerAlphabet(ProteinSequence.alphabet, 3)
        >>> kmer_codes = [kmer_alphabet.create_kmers(s.code) for s in sequences]
        >>> for code in kmer_codes:
        ...     print(code)
        [11701  4360  7879  9400  4419]
        [ 6517  4364  7975 11704  4419]
        >>> table = KmerTable.from_kmers(
        ...     kmer_alphabet, kmer_codes
        ... )
        >>> print(table)
        IQT: (0, 1)
        IQB: (1, 1)
        ITE: (0, 4), (1, 4)
        NIQ: (1, 0)
        QTI: (0, 2)
        QBI: (1, 2)
        TIT: (0, 3)
        BIQ: (0, 0)
        BIT: (1, 3)
        """
        _check_kmer_alphabet(kmer_alphabet)
        _check_multiple_kmer_bounds(kmers, kmer_alphabet)

        ref_ids = _compute_ref_ids(ref_ids, kmers)
        masks = _compute_masks(masks, kmers)

        table = KmerTable(kmer_alphabet)

        masks = [
            np.ones(len(arr), dtype=np.uint8) if mask is None
            # Convert boolean mask into uint8 array to be able
            # to handle it as memory view
            else np.frombuffer(
                mask.astype(bool, copy=False), dtype=np.uint8
            )
            for mask, arr in zip(masks, kmers)
        ]

        for arr, mask in zip(kmers, masks):
            table._count_masked_kmers(arr, mask)

        _init_c_arrays(table._ptr_array, EntrySize.NO_BUCKETS)

        for arr, ref_id, mask in zip(kmers, ref_ids, masks):
            table._add_kmers(arr, ref_id, mask)

        return table


    @staticmethod
    def from_kmer_selection(kmer_alphabet, positions, kmers, ref_ids=None):
        """
        from_kmer_selection(kmer_alphabet, positions, kmers, ref_ids=None)

        Create a :class:`KmerTable` by storing the positions of a
        filtered subset of input *k-mers*.

        This can be used to reduce the number of stored *k-mers* using
        a *k-mer* subset selector such as :class:`MinimizerSelector`.

        Parameters
        ----------
        kmer_alphabet : KmerAlphabet
            The :class:`KmerAlphabet` to use for the new table.
            Should be the same alphabet that was used to calculate the
            input *kmers*.
        positions : sized iterable object of (ndarray, shape=(n,), dtype=uint32), length=m
            List where each array contains the sequence positions of
            the filtered subset of *k-mers* given in `kmers`.
            The list may contain multiple elements for multiple
            sequences.
        kmers : sized iterable object of (ndarray, shape=(n,), dtype=np.int64), length=m
            List where each array contains the filtered subset of
            *k-mer* codes from a sequence.
            For each array the index of the *k-mer* code in the array,
            is stored in the table as sequence position.
            The list may contain multiple elements for multiple
            sequences.
        ref_ids : sized iterable object of int, length=m, optional
            The reference IDs for the sequences.
            These are used to identify the corresponding sequence for a
            *k-mer* match.
            By default the IDs are counted from *0* to *m*.

        Returns
        -------
        table : KmerTable
            The newly created table.

        Examples
        --------

        Reduce the size of sequence data in the table using minimizers:

        >>> sequence1 = ProteinSequence("THIS*IS*A*SEQVENCE")
        >>> kmer_alph = KmerAlphabet(sequence1.alphabet, k=3)
        >>> minimizer = MinimizerSelector(kmer_alph, window=4)
        >>> minimizer_pos, minimizers = minimizer.select(sequence1)
        >>> kmer_table = KmerTable.from_kmer_selection(
        ...     kmer_alph, [minimizer_pos], [minimizers]
        ... )

        Use the same :class:`MinimizerSelector` to select the minimizers
        from the query sequence and match them against the table.
        Although the amount of *k-mers* is reduced, matching is still
        guanrateed to work, if the two sequences share identity in the
        given window:

        >>> sequence2 = ProteinSequence("ANQTHER*SEQVENCE")
        >>> minimizer_pos, minimizers = minimizer.select(sequence2)
        >>> matches = kmer_table.match_kmer_selection(minimizer_pos, minimizers)
        >>> print(matches)
        [[ 9  0 11]
         [12  0 14]]
        >>> for query_pos, _, db_pos in matches:
        ...     print(sequence1)
        ...     print(" " * (db_pos-1) + "^" * kmer_table.k)
        ...     print(sequence2)
        ...     print(" " * (query_pos-1) + "^" * kmer_table.k)
        ...     print()
        THIS*IS*A*SEQVENCE
          ^^^
        ANQTHER*SEQVENCE
                ^^^
        <BLANKLINE>
        THIS*IS*A*SEQVENCE
                    ^^^
        ANQTHER*SEQVENCE
                ^^^
        <BLANKLINE>
        """
        _check_kmer_alphabet(kmer_alphabet)
        _check_multiple_kmer_bounds(kmers, kmer_alphabet)
        _check_position_shape(positions, kmers)

        ref_ids = _compute_ref_ids(ref_ids, kmers)

        table = KmerTable(kmer_alphabet)

        for arr in kmers:
            table._count_kmers(arr)

        _init_c_arrays(table._ptr_array, EntrySize.NO_BUCKETS)

        for pos, arr, ref_id in zip(positions, kmers, ref_ids):
            table._add_kmer_selection(
                pos.astype(np.uint32, copy=False), arr, ref_id
            )

        return table


    @staticmethod
    def from_tables(tables):
        """
        from_tables(tables)

        Create a :class:`KmerTable` by merging the *k-mer* positions
        from existing `tables`.

        Parameters
        ----------
        tables : iterable object of KmerTable
            The tables to be merged.
            All tables must have equal :class:`KmerAlphabet` objects,
            i.e. the same *k* and equal base alphabets.

        Returns
        -------
        table : KmerTable
            The newly created table.

        Examples
        --------

        >>> table1 = KmerTable.from_sequences(
        ...     2, [NucleotideSequence("TTATA")], ref_ids=[100]
        ... )
        >>> table2 = KmerTable.from_sequences(
        ...     2, [NucleotideSequence("CTAG")], ref_ids=[101]
        ... )
        >>> merged_table = KmerTable.from_tables([table1, table2])
        >>> print(merged_table)
        AG: (101, 2)
        AT: (100, 2)
        CT: (101, 0)
        TA: (100, 1), (100, 3), (101, 1)
        TT: (100, 0)
        """
        cdef KmerTable table

        _check_same_kmer_alphabet(tables)

        merged_table = KmerTable(tables[0].kmer_alphabet)

        # Sum the number of appearances of each k-mer from the tables
        for table in tables:
            # 'merged_table._ptr_array' is repurposed as count array,
            # This can be safely done, because in this step the pointers
            # are not initialized yet.
            # This may save a lot of memory because no extra array is
            # required to count the number of positions for each *k-mer*
            _count_table_entries(
                merged_table._ptr_array, table._ptr_array,
                EntrySize.NO_BUCKETS
            )

        _init_c_arrays(merged_table._ptr_array, EntrySize.NO_BUCKETS)

        for table in tables:
            _append_entries(merged_table._ptr_array, table._ptr_array)

        return merged_table


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    def from_positions(kmer_alphabet, dict kmer_positions):
        """
        from_positions(kmer_alphabet, kmer_positions)

        Create a :class:`KmerTable` from *k-mer* reference IDs and
        positions.
        This constructor is especially useful for restoring a table
        from previously serialized data.

        Parameters
        ----------
        kmer_alphabet : KmerAlphabet
            The :class:`KmerAlphabet` to use for the new table
        kmer_positions : dict of (int -> ndarray, shape=(n,2), dtype=int)
            A dictionary representing the *k-mer* reference IDs and
            positions to be stored in the newly created table.
            It maps a *k-mer* code to a :class:`ndarray`.
            To achieve a high performance the data type ``uint32``
            is preferred for the arrays.

        Returns
        -------
        table : KmerTable
            The newly created table.

        Examples
        --------

        >>> sequence = ProteinSequence("BIQTITE")
        >>> table = KmerTable.from_sequences(3, [sequence], ref_ids=[100])
        >>> print(table)
        IQT: (100, 1)
        ITE: (100, 4)
        QTI: (100, 2)
        TIT: (100, 3)
        BIQ: (100, 0)
        >>> data = {kmer: table[kmer] for kmer in table}
        >>> print(data)
        {4360: array([[100,   1]], dtype=uint32), 4419: array([[100,   4]], dtype=uint32), 7879: array([[100,   2]], dtype=uint32), 9400: array([[100,   3]], dtype=uint32), 11701: array([[100,   0]], dtype=uint32)}
        >>> restored_table = KmerTable.from_positions(table.kmer_alphabet, data)
        >>> print(restored_table)
        IQT: (100, 1)
        ITE: (100, 4)
        QTI: (100, 2)
        TIT: (100, 3)
        BIQ: (100, 0)
        """
        cdef int64 length
        cdef uint32* kmer_ptr
        cdef int64 i
        cdef int64 kmer
        cdef uint32[:,:] positions

        table = KmerTable(kmer_alphabet)

        cdef ptr[:] ptr_array = table._ptr_array
        cdef int64 alph_length = len(kmer_alphabet)

        for kmer, position_array in kmer_positions.items():
            if kmer < 0 or kmer >= alph_length:
                raise AlphabetError(
                    f"k-mer code {kmer} does not represent a valid k-mer"
                )
            positions = position_array.astype(np.uint32, copy=False)
            if positions.shape[0] == 0:
                # No position to add -> jump to the next k-mer
                continue
            if positions.shape[1] != 2:
                raise IndexError(
                    f"Each entry in position array has {positions.shape[1]} "
                    f"values, but 2 were expected"
                )

            # Plus the size of array length value (int64)
            length = 2 * positions.shape[0] + 2
            kmer_ptr = <uint32*>malloc(length * sizeof(uint32))
            if not kmer_ptr:
                raise MemoryError
            ptr_array[kmer] = <ptr>kmer_ptr
            (<int64*> kmer_ptr)[0] = length
            # Jump behind the length value
            kmer_ptr += 2

            # Add entries
            for i in range(positions.shape[0]):
                kmer_ptr[0] = positions[i,0]
                kmer_ptr += 1
                kmer_ptr[0] = positions[i,1]
                kmer_ptr += 1

        return table


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def match_table(self, KmerTable table, similarity_rule=None):
        """
        match_table(table, similarity_rule=None)

        Find matches between the *k-mers* in this table with the
        *k-mers* in another `table`.

        This means that for each *k-mer* the cartesian product between
        the positions in both tables is added to the matches.

        Parameters
        ----------
        table : KmerTable
            The table to be matched.
            Both tables must have equal :class:`KmerAlphabet` objects,
            i.e. the same *k* and equal base alphabets.
        similarity_rule : SimilarityRule, optional
            If this parameter is given, not only exact *k-mer* matches
            are considered, but also similar ones according to the given
            :class:`SimilarityRule`.

        Returns
        -------
        matches : ndarray, shape=(n,4), dtype=np.uint32
            The *k-mer* matches.
            Each row contains one match. Each match has the following
            columns:

                0. The reference ID of the matched sequence in the other
                   table
                1. The sequence position of the matched sequence in the
                   other table
                2. The reference ID of the matched sequence in this
                   table
                3. The sequence position of the matched sequence in this
                   table

        Notes
        -----

        There is no guaranteed order of the reference IDs or
        sequence positions in the returned matches.

        Examples
        --------

        >>> sequence1 = ProteinSequence("BIQTITE")
        >>> table1 = KmerTable.from_sequences(3, [sequence1], ref_ids=[100])
        >>> print(table1)
        IQT: (100, 1)
        ITE: (100, 4)
        QTI: (100, 2)
        TIT: (100, 3)
        BIQ: (100, 0)
        >>> sequence2 = ProteinSequence("TITANITE")
        >>> table2 = KmerTable.from_sequences(3, [sequence2], ref_ids=[101])
        >>> print(table2)
        ANI: (101, 3)
        ITA: (101, 1)
        ITE: (101, 5)
        NIT: (101, 4)
        TAN: (101, 2)
        TIT: (101, 0)
        >>> print(table1.match_table(table2))
        [[101   5 100   4]
         [101   0 100   3]]
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

        _check_same_kmer_alphabet((self, table))

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
                        # if-branch:
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
    def match(self, sequence, similarity_rule=None, ignore_mask=None):
        """
        match(sequence, similarity_rule=None, ignore_mask=None)

        Find matches between the *k-mers* in this table with all
        overlapping *k-mers* in the given `sequence`.
        *k* is determined by the table.

        Parameters
        ----------
        sequence : Sequence
            The sequence to be matched.
            The table's base alphabet must extend the alphabet of the
            sequence.
        similarity_rule : SimilarityRule, optional
            If this parameter is given, not only exact *k-mer* matches
            are considered, but also similar ones according to the given
            :class:`SimilarityRule`.
        ignore_mask : ndarray, dtype=bool, optional
            Boolean mask of sequence positions to ignore.
            *k-mers* that involve these sequence positions are not added
            to the table.
            This is used e.g. to skip repeat regions.
            By default, no sequence position is ignored.

        Returns
        -------
        matches : ndarray, shape=(n,3), dtype=np.uint32
            The *k-mer* matches.
            Each row contains one match. Each match has the following
            columns:

                0. The sequence position in the input sequence
                1. The reference ID of the matched sequence in the table
                2. The sequence position of the matched sequence in the
                   table

        Notes
        -----

        The matches are ordered by the first column.

        Examples
        --------

        >>> sequence1 = ProteinSequence("BIQTITE")
        >>> table = KmerTable.from_sequences(3, [sequence1], ref_ids=[100])
        >>> print(table)
        IQT: (100, 1)
        ITE: (100, 4)
        QTI: (100, 2)
        TIT: (100, 3)
        BIQ: (100, 0)
        >>> sequence2 = ProteinSequence("TITANITE")
        >>> print(table.match(sequence2))
        [[  0 100   3]
         [  5 100   4]]
        """
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
        cdef uint8[:] kmer_mask = _prepare_mask(
            self._kmer_alph, ignore_mask, len(sequence.code)
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
                        # if-branch:
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
    def match_kmer_selection(self, positions, kmers):
        """
        match_kmer_selection(positions, kmers)

        Find matches between the *k-mers* in this table with the given
        *k-mer* selection.

        It is intended to use this method to find matches in a table
        that was created using :meth:`from_kmer_selection()`.

        Parameters
        ----------
        positions : ndarray, shape=(n,), dtype=uint32
            Sequence positions of the filtered subset of *k-mers* given
            in `kmers`.
        kmers : ndarray, shape=(n,), dtype=np.int64
            Filtered subset of *k-mer* codes to match against.

        Returns
        -------
        matches : ndarray, shape=(n,3), dtype=np.uint32
            The *k-mer* matches.
            Each row contains one *k-mer* match.
            Each match has the following columns:

                0. The sequence position of the input *k-mer*, taken
                   from `positions`
                1. The reference ID of the matched sequence in the table
                2. The sequence position of the matched *k-mer* in the
                   table

        Examples
        --------

        Reduce the size of sequence data in the table using minimizers:

        >>> sequence1 = ProteinSequence("THIS*IS*A*SEQVENCE")
        >>> kmer_alph = KmerAlphabet(sequence1.alphabet, k=3)
        >>> minimizer = MinimizerSelector(kmer_alph, window=4)
        >>> minimizer_pos, minimizers = minimizer.select(sequence1)
        >>> kmer_table = KmerTable.from_kmer_selection(
        ...     kmer_alph, [minimizer_pos], [minimizers]
        ... )

        Use the same :class:`MinimizerSelector` to select the minimizers
        from the query sequence and match them against the table.
        Although the amount of *k-mers* is reduced, matching is still
        guanrateed to work, if the two sequences share identity in the
        given window:

        >>> sequence2 = ProteinSequence("ANQTHER*SEQVENCE")
        >>> minimizer_pos, minimizers = minimizer.select(sequence2)
        >>> matches = kmer_table.match_kmer_selection(minimizer_pos, minimizers)
        >>> print(matches)
        [[ 9  0 11]
         [12  0 14]]
        >>> for query_pos, _, db_pos in matches:
        ...     print(sequence1)
        ...     print(" " * (db_pos-1) + "^" * kmer_table.k)
        ...     print(sequence2)
        ...     print(" " * (query_pos-1) + "^" * kmer_table.k)
        ...     print()
        THIS*IS*A*SEQVENCE
          ^^^
        ANQTHER*SEQVENCE
                ^^^
        <BLANKLINE>
        THIS*IS*A*SEQVENCE
                    ^^^
        ANQTHER*SEQVENCE
                ^^^
        <BLANKLINE>
        """
        cdef int INIT_SIZE = 1

        cdef int64 i, j

        cdef int64 kmer
        cdef int64 match_i
        cdef int64 seq_pos
        cdef int64 length
        cdef uint32* kmer_ptr

        # Store in new variable
        # to disable repetitive initialization checks
        cdef ptr[:] ptr_array = self._ptr_array

        _check_kmer_bounds(kmers, self._kmer_alph)
        if positions.shape[0] != kmers.shape[0]:
            raise IndexError(
                f"{positions.shape[0]} positions were given "
                f"for {kmers.shape[0]} k-mers"
            )

        cdef uint32[:] pos_array = positions.astype(np.uint32, copy=False)
        cdef int64[:] kmer_array = kmers.astype(np.int64, copy=False)

        # This array will store the match positions
        # As the final number of matches is unknown, a list-like
        # approach is used:
        # The array is initialized with a relatively small inital size
        # and every time the limit would be exceeded its size is doubled
        cdef int64[:,:] matches = np.empty((INIT_SIZE, 3), dtype=np.int64)
        match_i = 0
        for i in range(kmer_array.shape[0]):
            kmer = kmer_array[i]
            seq_pos = pos_array[i]
            kmer_ptr = <uint32*>ptr_array[kmer]
            if kmer_ptr != NULL:
                # There is at least one entry for the k-mer
                length = (<int64*>kmer_ptr)[0]
                for j in range(2, length, 2):
                        if match_i >= matches.shape[0]:
                            # The 'matches' array is full
                            # -> double its size
                            matches = expand(np.asarray(matches))
                        matches[match_i, 0] = seq_pos
                        matches[match_i, 1] = kmer_ptr[j]
                        matches[match_i, 2] = kmer_ptr[j+1]
                        match_i += 1

        # Trim to correct size and return
        return np.asarray(matches[:match_i])


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def count(self, kmers=None):
        """
        count(kmers=None)

        Count the number of occurences for each *k-mer* in the table.

        Parameters
        ----------
        kmers : ndarray, dtype=np.int64, optional
            The count is returned for these *k-mer* codes.
            By default all *k-mers* are counted in ascending order, i.e.
            ``count_for_kmer = counts[kmer]``.

        Returns
        -------
        counts : ndarray, dtype=np.int64, optional
            The counts for each given *k-mer*.

        Examples
        --------
        >>> table = KmerTable.from_sequences(
        ...     k = 2,
        ...     sequences = [NucleotideSequence("TTATA"), NucleotideSequence("CTAG")],
        ...     ref_ids = [0, 1]
        ... )
        >>> print(table)
        AG: (1, 2)
        AT: (0, 2)
        CT: (1, 0)
        TA: (0, 1), (0, 3), (1, 1)
        TT: (0, 0)

        Count two selected *k-mers*:

        >>> print(table.count(table.kmer_alphabet.encode_multiple(["TA", "AG"])))
        [3 1]

        Count all *k-mers*:

        >>> counts = table.count()
        >>> print(counts)
        [0 0 1 1 0 0 0 1 0 0 0 0 3 0 0 1]
        >>> for kmer, count in zip(table.kmer_alphabet.get_symbols(), counts):
        ...     print(kmer, count)
        AA 0
        AC 0
        AG 1
        AT 1
        CA 0
        CC 0
        CG 0
        CT 1
        GA 0
        GC 0
        GG 0
        GT 0
        TA 3
        TC 0
        TG 0
        TT 1
        """
        cdef int64 i

        cdef int64 length
        cdef int64 kmer
        cdef int64* kmer_ptr
        cdef ptr[:] ptr_array = self._ptr_array
        cdef int64[:] kmer_array
        cdef int64[:] counts

        if kmers is None:
            counts = np.zeros(ptr_array.shape[0], dtype=np.int64)
            for kmer in range(ptr_array.shape[0]):
                kmer_ptr = <int64*> (ptr_array[kmer])
                if kmer_ptr != NULL:
                    # First 64 bytes are length of C-array
                    length = kmer_ptr[0]
                    # Array length is measured in uint32
                    # length = 2 * count + 2 -> rearrange formula
                    counts[kmer] = (length - 2) // 2

        else:
            _check_kmer_bounds(kmers, self._kmer_alph)

            kmer_array = kmers.astype(np.int64, copy=False)
            counts = np.zeros(kmer_array.shape[0], dtype=np.int64)
            for i in range(kmer_array.shape[0]):
                kmer = kmer_array[i]
                kmer_ptr = <int64*> (ptr_array[kmer])
                if kmer_ptr != NULL:
                    length = kmer_ptr[0]
                    counts[i] = (length - 2) // 2

        return np.asarray(counts)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_kmers(self):
        """
        Get the *k-mer* codes for all *k-mers* that have at least one
        position in the table.

        Returns
        -------
        kmers : ndarray, shape=(n,), dtype=np.int64
            The *k-mer* codes.

        Examples
        --------

        >>> sequence = ProteinSequence("BIQTITE")
        >>> table = KmerTable.from_sequences(3, [sequence], ref_ids=[100])
        >>> print(table)
        IQT: (100, 1)
        ITE: (100, 4)
        QTI: (100, 2)
        TIT: (100, 3)
        BIQ: (100, 0)
        >>> kmer_codes = table.get_kmers()
        >>> print(kmer_codes)
        [ 4360  4419  7879  9400 11701]
        >>> for code in kmer_codes:
        ...     print(table[code])
        [[100   1]]
        [[100   4]]
        [[100   2]]
        [[100   3]]
        [[100   0]]
        """
        cdef int64 kmer
        cdef ptr[:] ptr_array = self._ptr_array

        # Pessimistic allocation:
        # The maximum number of used kmers are all possible kmers
        cdef int64[:] kmers = np.zeros(ptr_array.shape[0], dtype=np.int64)

        cdef int64 i = 0
        for kmer in range(ptr_array.shape[0]):
            if <uint32*> (ptr_array[kmer]) != NULL:
                kmers[i] = kmer
                i += 1

        # Trim to correct size
        return np.asarray(kmers)[:i]


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __getitem__(self, int64 kmer):
        cdef int64 i, j
        cdef int64 length
        cdef uint32* kmer_ptr
        cdef uint32[:,:] positions

        if kmer >= len(self):
            raise AlphabetError(
                f"k-mer code {kmer} is out of bounds "
                f"for the given KmerAlphabet"
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


    def __len__(self):
        return len(self._kmer_alph)


    def __contains__(self, int64 kmer):
        # If there is at least one entry for a k-mer,
        # the pointer is not NULL
        return self._ptr_array[kmer] != 0


    def __iter__(self):
        for kmer in self.get_kmers():
            yield kmer.item()


    def __reversed__(self):
        return reversed(self.get_kmers())


    def __eq__(self, item):
        if item is self:
            return True
        if type(item) != KmerTable:
            return False

        # Introduce static typing to access statically typed fields
        cdef KmerTable other = item
        if self._kmer_alph.base_alphabet != other._kmer_alph.base_alphabet:
            return False
        if self._k != other._k:
            return False
        return _equal_c_arrays(self._ptr_array, other._ptr_array)


    def __str__(self):
        return _to_string(self)


    def __getnewargs_ex__(self):
        return (self._kmer_alph,), {}


    def __getstate__(self):
        return _pickle_c_arrays(self._ptr_array)


    def __setstate__(self, state):
        _unpickle_c_arrays(self._ptr_array, state)


    def __dealloc__(self):
        if self._is_initialized():
            _deallocate_ptrs(self._ptr_array)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _count_kmers(self, int64[:] kmers):
        """
        Repurpose the pointer array as count array and add the
        total number of positions for the given kmers to the values in
        the count array.

        This can be safely done, because in this step the pointers are
        not initialized yet.
        This may save a lot of memory because no extra array is required
        to count the number of positions for each *k-mer*.
        """
        cdef uint32 seq_pos
        cdef int64 kmer

        cdef ptr[:] count_array = self._ptr_array

        for seq_pos in range(kmers.shape[0]):
            kmer = kmers[seq_pos]
            count_array[kmer] += 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _count_masked_kmers(self, int64[:] kmers, uint8[:] mask):
        """
        Same as above, but with mask.
        """
        cdef uint32 seq_pos
        cdef int64 kmer

        cdef ptr[:] count_array = self._ptr_array

        for seq_pos in range(kmers.shape[0]):
            if mask[seq_pos]:
                kmer = kmers[seq_pos]
                count_array[kmer] += 1


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_kmers(self, int64[:] kmers, uint32 ref_id, uint8[:] mask):
        """
        For each *k-mer* in `kmers` add the reference ID and the
        position in the array to the corresponding C-array and update
        the length of the C-array.
        """
        cdef uint32 seq_pos
        cdef int64 current_size
        cdef int64 kmer
        cdef uint32* kmer_ptr

        # Store in new variable
        # to disable repetitive initialization checks
        cdef ptr[:] ptr_array = self._ptr_array

        if mask.shape[0] != kmers.shape[0]:
            raise IndexError(
                f"Mask has length {mask.shape[0]}, "
                f"but there are {kmers.shape[0]} k-mers"
            )

        for seq_pos in range(kmers.shape[0]):
            if mask[seq_pos]:
                kmer = kmers[seq_pos]
                kmer_ptr = <uint32*> ptr_array[kmer]

                # Append k-mer reference ID and position
                current_size = (<int64*> kmer_ptr)[0]
                kmer_ptr[current_size    ] = ref_id
                kmer_ptr[current_size + 1] = seq_pos
                (<int64*> kmer_ptr)[0] = current_size + EntrySize.NO_BUCKETS

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_kmer_selection(self, uint32[:] positions, int64[:] kmers,
                         uint32 ref_id):
        """
        For each *k-mer* in `kmers` add the reference ID and the
        position from `positions` to the corresponding C-array and
        update the length of the C-array.
        """
        cdef uint32 i
        cdef uint32 seq_pos
        cdef int64 current_size
        cdef int64 kmer
        cdef uint32* kmer_ptr

        if positions.shape[0] != kmers.shape[0]:
            raise IndexError(
                f"{positions.shape[0]} positions were given "
                f"for {kmers.shape[0]} k-mers"
            )

        # Store in new variable
        # to disable repetitive initialization checks
        cdef ptr[:] ptr_array = self._ptr_array

        for i in range(positions.shape[0]):
            kmer = kmers[i]
            seq_pos = positions[i]
            kmer_ptr = <uint32*> ptr_array[kmer]

            # Append k-mer reference ID and position
            current_size = (<int64*> kmer_ptr)[0]
            kmer_ptr[current_size    ] = ref_id
            kmer_ptr[current_size + 1] = seq_pos
            (<int64*> kmer_ptr)[0] = current_size + EntrySize.NO_BUCKETS


    cdef inline bint _is_initialized(self):
        # Memoryviews are not initialized on class creation
        # This method checks, if the _ptr_array memoryview was
        # initialized and is not None
        try:
            if self._ptr_array is not None:
                return True
            else:
                return False
        except AttributeError:
            return False




cdef class BucketKmerTable:
    """
    This class represents a *k-mer* index table.
    In contrast to :class:`KmerTable` it does store each unique *k-mer*
    in a separate C-array, but limits the number of C-arrays instead
    to a number of buckets.
    Hence, different *k-mer* may be stored in the same bucket, like in a
    hash table.
    This approach makes *k-mer* indices with large *k-mer* alphabets
    fit into memory.

    Otherwise, the API for creating a :class:`BucketKmerTable` and
    matching to it is analogous to :class:`KmerTable`.

    Attributes
    ----------
    kmer_alphabet : KmerAlphabet
        The internal :class:`KmerAlphabet`, that is used to
        encode all overlapping *k-mers* of an input sequence.
    alphabet : Alphabet
        The base alphabet, from which this :class:`BucketKmerTable` was
        created.
    k : int
        The length of the *k-mers*.
    n_buckets : int
        The number of buckets, the *k-mers* are divided into.

    See Also
    --------
    KmerTable

    Notes
    -----

    *Memory consumption*

    For efficient mapping, a :class:`BucketKmerTable` contains a pointer
    array, that contains one 64-bit pointer for each bucket.
    If there is at least one position for a bucket, the corresponding
    pointer points to a C-array that contains

        1. The length of the C-array *(int64)*
        2. The *k-mers* *(int64)*
        3. The reference ID for each *k-mer* *(uint32)*
        4. The sequence position for each *k-mer* *(uint32)*

    As buckets are used, the memory requirements are limited to the number
    of buckets instead of scaling with the :class:`KmerAlphabet` size.
    If each bucket is used, the required memory space :math:`S` in byte
    is

    .. math::

        S = 16B + 16L

    where :math:`B` is the number of buckets and :math:`L` is the summed
    length of all sequences added to the table.

    *Buckets*

    The ratio :math:`L/B` is called *load_factor*.
    By default :class:`BucketKmerTable` uses a load factor of
    approximately 0.8 to ensure efficient *k-mer* matching.
    The number fo buckets can be adjusted by setting the
    `n_buckets` parameters on :class:`BucketKmerTable` creation.
    It is recommended to use :func:`bucket_number()` to compute an
    appropriate number of buckets.

    *Multiprocessing*

    :class:`BucketKmerTable` objects can be used in multi-processed
    setups:
    Adding a large database of sequences to a table can be sped up by
    splitting the database into smaller chunks and create a separate
    table for each chunk in separate processes.
    Eventually, the tables can be merged to one large table using
    :meth:`from_tables()`.

    Since :class:`BucketKmerTable` supports the *pickle* protocol,
    the matching step can also be divided into multiple processes, if
    multiple sequences need to be matched.

    *Storage on hard drive*

    The most time efficient way to read/write a :class:`BucketKmerTable`
    is the *pickle* format.

    *Indexing and iteration*

    Due to the higher complexity in the *k-mer* lookup compared to
    :class:`KmerTable`, this class is still indexable but not iterable.

    Examples
    --------

    Create a *2-mer* index table for some nucleotide sequences:

    >>> table = BucketKmerTable.from_sequences(
    ...     k = 2,
    ...     sequences = [NucleotideSequence("TTATA"), NucleotideSequence("CTAG")],
    ...     ref_ids = [0, 1]
    ... )

    Display the contents of the table as
    (reference ID, sequence position) tuples:

    >>> print(table)
    AG: (1, 2)
    AT: (0, 2)
    CT: (1, 0)
    TA: (0, 1), (0, 3), (1, 1)
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
    """

    cdef object _kmer_alph
    cdef int _k
    cdef int64 _n_buckets

    # The pointer array is the core of the index table:
    # It maps each possible k-mer bucket (represented by its code) to a
    # C-array of indices.
    # Each entry in a C-array contains the k-mer code, a reference ID
    # and the location in that sequence where that k-mer appears
    # The memory layout of each C-array is as following:
    #
    # (Array length) (k-mer 0)  (RefID 0)   (Position 0) (k-mer 1) ...
    # -----int64----|--int64--|---uint32---|---uint32---|--int64--
    #
    # The array length is based on 32 bit units.
    # If there is no entry for a k-mer bucket, the respective pointer is
    # NULL.
    cdef ptr[:] _ptr_array


    def __cinit__(self, n_buckets, kmer_alphabet):
        # This check is necessary for proper memory management
        # of the allocated arrays
        if self._is_initialized():
            raise Exception("Duplicate call of constructor")

        self._kmer_alph = kmer_alphabet
        self._k = kmer_alphabet.k
        if len(self._kmer_alph) < n_buckets:
            self._n_buckets = len(self._kmer_alph)
        else:
            self._n_buckets = n_buckets
        self._ptr_array = np.zeros(self._n_buckets, dtype=np.uint64)


    @property
    def kmer_alphabet(self):
        return self._kmer_alph

    @property
    def alphabet(self):
        return self._kmer_alph.base_alphabet

    @property
    def k(self):
        return self._k

    @property
    def n_buckets(self):
        return self._n_buckets

    @staticmethod
    def from_sequences(k, sequences, ref_ids=None, ignore_masks=None,
                       alphabet=None, spacing=None, n_buckets=None):
        """
        from_sequences(k, sequences, ref_ids=None, ignore_masks=None,
                       alphabet=None, spacing=None, n_buckets=None)

        Create a :class:`BucketKmerTable` by storing the positions of
        all overlapping *k-mers* from the input `sequences`.

        Parameters
        ----------
        k : int
            The length of the *k-mers*.
        sequences : sized iterable object of Sequence, length=m
            The sequences to get the *k-mer* positions from.
            These sequences must have equal alphabets, or one of these
            sequences must have an alphabet that extends the alphabets
            of all other sequences.
        ref_ids : sized iterable object of int, length=m, optional
            The reference IDs for the given sequences.
            These are used to identify the corresponding sequence for a
            *k-mer* match.
            By default the IDs are counted from *0* to *m*.
        ignore_masks : sized iterable object of (ndarray, dtype=bool), length=m, optional
            Sequence positions to ignore.
            *k-mers* that involve these sequence positions are not added
            to the table.
            This is used e.g. to skip repeat regions.
            If provided, the list must contain one boolean mask
            (or ``None``) for each sequence, and each bolean mask must
            have the same length as the sequence.
            By default, no sequence position is ignored.
        alphabet : Alphabet, optional
            The alphabet to use for this table.
            It must extend the alphabets of the input `sequences`.
            By default, an appropriate alphabet is inferred from the
            input `sequences`.
            This option is usually used for compatibility with another
            sequence/table in the matching step.
        spacing : None or str or list or ndarray, dtype=int, shape=(k,)
            If provided, spaced *k-mers* are used instead of continuous
            ones.
            The value contains the *informative* positions relative to
            the start of the *k-mer*, also called the *model*.
            The number of *informative* positions must equal *k*.
            Refer to :class:`KmerAlphabet` for more details.
        n_buckets : int, optional
            Set the number of buckets in the table, e.g. to use a
            different load factor.
            It is recommended to use :func:`bucket_number()` for this
            purpose.
            By default, a load factor of approximately 0.8 is used.

        See Also
        --------
        from_kmers : The same functionality based on already created *k-mers*

        Returns
        -------
        table : BucketKmerTable
            The newly created table.

        Examples
        --------

        >>> sequences = [NucleotideSequence("TTATA"), NucleotideSequence("CTAG")]
        >>> table = BucketKmerTable.from_sequences(
        ...     2, sequences, ref_ids=[100, 101]
        ... )
        >>> print(table)
        AG: (101, 2)
        AT: (100, 2)
        CT: (101, 0)
        TA: (100, 1), (100, 3), (101, 1)
        TT: (100, 0)

        Give an explicit compatible alphabet:

        >>> table = BucketKmerTable.from_sequences(
        ...     2, sequences, ref_ids=[100, 101],
        ...     alphabet=NucleotideSequence.ambiguous_alphabet()
        ... )

        Ignore all ``N`` in a sequence:

        >>> sequence = NucleotideSequence("ACCNTANNG")
        >>> table = BucketKmerTable.from_sequences(
        ...     2, [sequence], ignore_masks=[sequence.symbols == "N"]
        ... )
        >>> print(table)
        AC: (0, 0)
        CC: (0, 1)
        TA: (0, 4)
        """
        ref_ids = _compute_ref_ids(ref_ids, sequences)
        ignore_masks = _compute_masks(ignore_masks, sequences)
        alphabet = _compute_alphabet(
            alphabet, (sequence.alphabet for sequence in sequences)
        )
        kmer_alphabet = KmerAlphabet(alphabet, k, spacing)

        # Calculate k-mers
        kmers_list = [
            kmer_alphabet.create_kmers(sequence.code)
            for sequence in sequences
        ]

        if n_buckets is None:
            n_kmers = np.sum([len(kmers) for kmers in kmers_list])
            n_buckets = bucket_number(n_kmers)

        table = BucketKmerTable(n_buckets, kmer_alphabet)

        masks = [
            _prepare_mask(kmer_alphabet, ignore_mask, len(sequence))
            for sequence, ignore_mask in zip(sequences, ignore_masks)
        ]

        # Count the number of appearances of each k-mer and store the
        # result in the pointer array, that is now used as count array
        for kmers, mask in zip(kmers_list, masks):
            table._count_masked_kmers(kmers, mask)

        # Transfrom count array into pointer array with C-array of
        # appropriate size
        _init_c_arrays(table._ptr_array, EntrySize.BUCKETS)

        # Fill the C-arrays with the k-mer positions
        for kmers, ref_id, mask in zip(kmers_list, ref_ids, masks):
            table._add_kmers(kmers, ref_id, mask)

        return table


    @staticmethod
    def from_kmers(kmer_alphabet, kmers, ref_ids=None, masks=None,
                   n_buckets=None):
        """
        from_kmers(kmer_alphabet, kmers, ref_ids=None, masks=None,
                   n_buckets=None)

        Create a :class:`BucketKmerTable` by storing the positions of
        all input *k-mers*.

        Parameters
        ----------
        kmer_alphabet : KmerAlphabet
            The :class:`KmerAlphabet` to use for the new table.
            Should be the same alphabet that was used to calculate the
            input *kmers*.
        kmers : sized iterable object of (ndarray, dtype=np.int64), length=m
            List where each array contains the *k-mer* codes from a
            sequence.
            For each array the index of the *k-mer* code in the array
            is stored in the table as sequence position.
        ref_ids : sized iterable object of int, length=m, optional
            The reference IDs for the sequences.
            These are used to identify the corresponding sequence for a
            *k-mer* match.
            By default the IDs are counted from *0* to *m*.
        masks : sized iterable object of (ndarray, dtype=bool), length=m, optional
            A *k-mer* code at a position, where the corresponding mask
            is false, is not added to the table.
            By default, all positions are added.
        n_buckets : int, optional
            Set the number of buckets in the table, e.g. to use a
            different load factor.
            It is recommended to use :func:`bucket_number()` for this
            purpose.
            By default, a load factor of approximately 0.8 is used.

        See Also
        --------
        from_sequences : The same functionality based on undecomposed sequences

        Returns
        -------
        table : BucketKmerTable
            The newly created table.

        Examples
        --------

        >>> sequences = [ProteinSequence("BIQTITE"), ProteinSequence("NIQBITE")]
        >>> kmer_alphabet = KmerAlphabet(ProteinSequence.alphabet, 3)
        >>> kmer_codes = [kmer_alphabet.create_kmers(s.code) for s in sequences]
        >>> for code in kmer_codes:
        ...     print(code)
        [11701  4360  7879  9400  4419]
        [ 6517  4364  7975 11704  4419]
        >>> table = BucketKmerTable.from_kmers(kmer_alphabet, kmer_codes)
        >>> print(table)
        IQT: (0, 1)
        IQB: (1, 1)
        ITE: (0, 4), (1, 4)
        NIQ: (1, 0)
        QTI: (0, 2)
        QBI: (1, 2)
        TIT: (0, 3)
        BIQ: (0, 0)
        BIT: (1, 3)
        """
        _check_kmer_alphabet(kmer_alphabet)
        _check_multiple_kmer_bounds(kmers, kmer_alphabet)

        ref_ids = _compute_ref_ids(ref_ids, kmers)
        masks = _compute_masks(masks, kmers)

        if n_buckets is None:
            n_kmers = np.sum([len(e) for e in kmers])
            n_buckets = bucket_number(n_kmers)

        table = BucketKmerTable(n_buckets, kmer_alphabet)

        masks = [
            np.ones(len(arr), dtype=np.uint8) if mask is None
            # Convert boolean mask into uint8 array to be able
            # to handle it as memory view
            else np.frombuffer(
                mask.astype(bool, copy=False), dtype=np.uint8
            )
            for mask, arr in zip(masks, kmers)
        ]

        for arr, mask in zip(kmers, masks):
            table._count_masked_kmers(arr, mask)

        _init_c_arrays(table._ptr_array, EntrySize.BUCKETS)

        for arr, ref_id, mask in zip(kmers, ref_ids, masks):
            table._add_kmers(arr, ref_id, mask)

        return table


    @staticmethod
    def from_kmer_selection(kmer_alphabet, positions, kmers, ref_ids=None,
                            n_buckets=None):
        """
        from_kmer_selection(kmer_alphabet, positions, kmers, ref_ids=None,
                            n_buckets=None)

        Create a :class:`BucketKmerTable` by storing the positions of a
        filtered subset of input *k-mers*.

        This can be used to reduce the number of stored *k-mers* using
        a *k-mer* subset selector such as :class:`MinimizerSelector`.

        Parameters
        ----------
        kmer_alphabet : KmerAlphabet
            The :class:`KmerAlphabet` to use for the new table.
            Should be the same alphabet that was used to calculate the
            input *kmers*.
        positions : sized iterable object of (ndarray, shape=(n,), dtype=uint32), length=m
            List where each array contains the sequence positions of
            the filtered subset of *k-mers* given in `kmers`.
            The list may contain multiple elements for multiple
            sequences.
        kmers : sized iterable object of (ndarray, shape=(n,), dtype=np.int64), length=m
            List where each array contains the filtered subset of
            *k-mer* codes from a sequence.
            For each array the index of the *k-mer* code in the array,
            is stored in the table as sequence position.
            The list may contain multiple elements for multiple
            sequences.
        ref_ids : sized iterable object of int, length=m, optional
            The reference IDs for the sequences.
            These are used to identify the corresponding sequence for a
            *k-mer* match.
            By default the IDs are counted from *0* to *m*.
        n_buckets : int, optional
            Set the number of buckets in the table, e.g. to use a
            different load factor.
            It is recommended to use :func:`bucket_number()` for this
            purpose.
            By default, a load factor of approximately 0.8 is used.

        Returns
        -------
        table : BucketKmerTable
            The newly created table.

        Examples
        --------

        Reduce the size of sequence data in the table using minimizers:

        >>> sequence1 = ProteinSequence("THIS*IS*A*SEQVENCE")
        >>> kmer_alph = KmerAlphabet(sequence1.alphabet, k=3)
        >>> minimizer = MinimizerSelector(kmer_alph, window=4)
        >>> minimizer_pos, minimizers = minimizer.select(sequence1)
        >>> kmer_table = BucketKmerTable.from_kmer_selection(
        ...     kmer_alph, [minimizer_pos], [minimizers]
        ... )

        Use the same :class:`MinimizerSelector` to select the minimizers
        from the query sequence and match them against the table.
        Although the amount of *k-mers* is reduced, matching is still
        guanrateed to work, if the two sequences share identity in the
        given window:

        >>> sequence2 = ProteinSequence("ANQTHER*SEQVENCE")
        >>> minimizer_pos, minimizers = minimizer.select(sequence2)
        >>> matches = kmer_table.match_kmer_selection(minimizer_pos, minimizers)
        >>> print(matches)
        [[ 9  0 11]
         [12  0 14]]
        >>> for query_pos, _, db_pos in matches:
        ...     print(sequence1)
        ...     print(" " * (db_pos-1) + "^" * kmer_table.k)
        ...     print(sequence2)
        ...     print(" " * (query_pos-1) + "^" * kmer_table.k)
        ...     print()
        THIS*IS*A*SEQVENCE
          ^^^
        ANQTHER*SEQVENCE
                ^^^
        <BLANKLINE>
        THIS*IS*A*SEQVENCE
                    ^^^
        ANQTHER*SEQVENCE
                ^^^
        <BLANKLINE>
        """
        _check_kmer_alphabet(kmer_alphabet)
        _check_multiple_kmer_bounds(kmers, kmer_alphabet)
        _check_position_shape(positions, kmers)

        ref_ids = _compute_ref_ids(ref_ids, kmers)

        if n_buckets is None:
            n_kmers = np.sum([len(e) for e in kmers])
            n_buckets = bucket_number(n_kmers)

        table = BucketKmerTable(n_buckets, kmer_alphabet)

        for arr in kmers:
            table._count_kmers(arr)

        _init_c_arrays(table._ptr_array, EntrySize.BUCKETS)

        for pos, arr, ref_id in zip(positions, kmers, ref_ids):
            table._add_kmer_selection(
                pos.astype(np.uint32, copy=False), arr, ref_id
            )

        return table


    @staticmethod
    def from_tables(tables):
        """
        from_tables(tables)

        Create a :class:`BucketKmerTable` by merging the *k-mer*
        positions from existing `tables`.

        Parameters
        ----------
        tables : iterable object of BucketKmerTable
            The tables to be merged.
            All tables must have equal number of buckets and equal
            :class:`KmerAlphabet` objects, i.e. the same *k* and equal
            base alphabets.

        Returns
        -------
        table : BucketKmerTable
            The newly created table.

        Examples
        --------
        To ensure that all tables have the same number of buckets,
        `n_buckets` need to be set on table creation.

        >>> # The sequence length is not exactly the length of resulting k-mers,
        >>> # but it is close enough for bucket computation
        >>> n_buckets = bucket_number(len("TTATA") + len("CTAG"))
        >>> table1 = BucketKmerTable.from_sequences(
        ...     2, [NucleotideSequence("TTATA")], ref_ids=[100],
        ...     n_buckets=n_buckets
        ... )
        >>> table2 = BucketKmerTable.from_sequences(
        ...     2, [NucleotideSequence("CTAG")], ref_ids=[101],
        ...     n_buckets=n_buckets
        ... )
        >>> merged_table = BucketKmerTable.from_tables([table1, table2])
        >>> print(merged_table)
        AG: (101, 2)
        AT: (100, 2)
        CT: (101, 0)
        TA: (100, 1), (100, 3), (101, 1)
        TT: (100, 0)
        """
        cdef BucketKmerTable table

        _check_same_kmer_alphabet(tables)
        _check_same_buckets(tables)

        merged_table = BucketKmerTable(
            tables[0].n_buckets,
            tables[0].kmer_alphabet
        )

        # Sum the number of appearances of each k-mer from the tables
        for table in tables:
            _count_table_entries(
                merged_table._ptr_array, table._ptr_array,
                EntrySize.BUCKETS
            )

        _init_c_arrays(merged_table._ptr_array, EntrySize.BUCKETS)

        for table in tables:
            _append_entries(merged_table._ptr_array, table._ptr_array)

        return merged_table


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def match_table(self, BucketKmerTable table, similarity_rule=None):
        """
        match_table(table, similarity_rule=None)

        Find matches between the *k-mers* in this table with the
        *k-mers* in another `table`.

        This means that for each *k-mer* the cartesian product between
        the positions in both tables is added to the matches.

        Parameters
        ----------
        table : BucketKmerTable
            The table to be matched.
            Both tables must have equal number of buckets and equal
            :class:`KmerAlphabet` objects, i.e. the same *k* and equal
            base alphabets.
        similarity_rule : SimilarityRule, optional
            If this parameter is given, not only exact *k-mer* matches
            are considered, but also similar ones according to the given
            :class:`SimilarityRule`.

        Returns
        -------
        matches : ndarray, shape=(n,4), dtype=np.uint32
            The *k-mer* matches.
            Each row contains one match. Each match has the following
            columns:

                0. The reference ID of the matched sequence in the other
                   table
                1. The sequence position of the matched sequence in the
                   other table
                2. The reference ID of the matched sequence in this
                   table
                3. The sequence position of the matched sequence in this
                   table

        Notes
        -----


        There is no guaranteed order of the reference IDs or
        sequence positions in the returned matches.

        Examples
        --------
        To ensure that both tables have the same number of buckets,
        `n_buckets` need to be set on table creation.

        >>> # The sequence length is not exactly the length of resulting k-mers,
        >>> # but it is close enouggh for bucket computation
        >>> n_buckets = bucket_number(max(len("BIQTITE"), len("TITANITE")))
        >>> sequence1 = ProteinSequence("BIQTITE")
        >>> table1 = BucketKmerTable.from_sequences(3, [sequence1], ref_ids=[100])
        >>> print(table1)
        IQT: (100, 1)
        ITE: (100, 4)
        QTI: (100, 2)
        TIT: (100, 3)
        BIQ: (100, 0)
        >>> sequence2 = ProteinSequence("TITANITE")
        >>> table2 = BucketKmerTable.from_sequences(3, [sequence2], ref_ids=[101])
        >>> print(table2)
        ANI: (101, 3)
        ITA: (101, 1)
        ITE: (101, 5)
        NIT: (101, 4)
        TAN: (101, 2)
        TIT: (101, 0)
        >>> print(table1.match_table(table2))
        [[101   0 100   3]
         [101   5 100   4]]
        """
        cdef int INIT_SIZE = 1

        cdef int64 bucket, sim_bucket
        cdef int64 self_kmer, other_kmer, sim_kmer
        cdef int64 match_i
        cdef int64 i, j, l
        cdef int64 self_length, other_length
        cdef uint32* self_bucket_ptr
        cdef uint32* other_bucket_ptr

        # This variable will only be used if a similarity rule exists
        cdef int64[:] similar_kmers

        # Store in new variables
        # to disable repetitive initialization checks
        cdef ptr[:] self_ptr_array = self._ptr_array
        cdef ptr[:] other_ptr_array = table._ptr_array

        _check_same_kmer_alphabet((self, table))
        _check_same_buckets((self, table))

        # This array will store the match positions
        # As the final number of matches is unknown, a list-like
        # approach is used:
        # The array is initialized with a relatively small inital size
        # and every time the limit would be exceeded its size is doubled
        cdef int64[:,:] matches = np.empty((INIT_SIZE, 4), dtype=np.int64)
        match_i = 0
        if similarity_rule is None:
            for bucket in range(self_ptr_array.shape[0]):
                self_bucket_ptr = <uint32*>self_ptr_array[bucket]
                other_bucket_ptr = <uint32*>other_ptr_array[bucket]
                if self_bucket_ptr != NULL and other_bucket_ptr != NULL:
                    # This bucket exists for both tables
                    other_length = (<int64*>other_bucket_ptr)[0]
                    self_length  = (<int64*>self_bucket_ptr )[0]
                    for i in range(2, other_length, 4):
                        # Hacky syntax to achieve casting to int64*
                        # after offset is applied
                        other_kmer = (<int64*>(other_bucket_ptr + i))[0]
                        for j in range(2, self_length, 4):
                            self_kmer = (<int64*>(self_bucket_ptr + j))[0]
                            if self_kmer == other_kmer:
                                # The k-mers are not only in the same
                                # bucket, but they are actually equal
                                if match_i >= matches.shape[0]:
                                    # The 'matches' array is full
                                    # -> double its size
                                    matches = expand(np.asarray(matches))
                                matches[match_i, 0] = other_bucket_ptr[i+2]
                                matches[match_i, 1] = other_bucket_ptr[i+3]
                                matches[match_i, 2] = self_bucket_ptr[j+2]
                                matches[match_i, 3] = self_bucket_ptr[j+3]
                                match_i += 1

        else:
            for bucket in range(self_ptr_array.shape[0]):
                other_bucket_ptr = <uint32*>other_ptr_array[bucket]
                if other_bucket_ptr != NULL:
                    other_length = (<int64*>other_bucket_ptr)[0]
                    for i in range(2, other_length, 4):
                        other_kmer = (<int64*>(other_bucket_ptr + i))[0]
                        # If a similarity rule exists, iterate not only over
                        # the exact k-mer, but over all k-mers similar to
                        # the current k-mer
                        similar_kmers = similarity_rule.similar_kmers(
                            self._kmer_alph, other_kmer
                        )
                        for l in range(similar_kmers.shape[0]):
                            sim_kmer = similar_kmers[l]
                            sim_bucket = sim_kmer % self._n_buckets
                            self_bucket_ptr = <uint32*>self_ptr_array[sim_bucket]
                            if self_bucket_ptr != NULL:
                                self_length  = (<int64*>self_bucket_ptr)[0]
                                for j in range(2, self_length, 4):
                                    self_kmer = (<int64*>(self_bucket_ptr + j))[0]
                                    if self_kmer == sim_kmer:
                                        if match_i >= matches.shape[0]:
                                            # The 'matches' array is full
                                            # -> double its size
                                            matches = expand(np.asarray(matches))
                                        matches[match_i, 0] = other_bucket_ptr[i+2]
                                        matches[match_i, 1] = other_bucket_ptr[i+3]
                                        matches[match_i, 2] = self_bucket_ptr[j+2]
                                        matches[match_i, 3] = self_bucket_ptr[j+3]
                                        match_i += 1

        # Trim to correct size and return
        return np.asarray(matches[:match_i])


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def match(self, sequence, similarity_rule=None, ignore_mask=None):
        """
        match(sequence, similarity_rule=None, ignore_mask=None)

        Find matches between the *k-mers* in this table with all
        overlapping *k-mers* in the given `sequence`.
        *k* is determined by the table.

        Parameters
        ----------
        sequence : Sequence
            The sequence to be matched.
            The table's base alphabet must extend the alphabet of the
            sequence.
        similarity_rule : SimilarityRule, optional
            If this parameter is given, not only exact *k-mer* matches
            are considered, but also similar ones according to the given
            :class:`SimilarityRule`.
        ignore_mask : ndarray, dtype=bool, optional
            Boolean mask of sequence positions to ignore.
            *k-mers* that involve these sequence positions are not added
            to the table.
            This is used e.g. to skip repeat regions.
            By default, no sequence position is ignored.

        Returns
        -------
        matches : ndarray, shape=(n,3), dtype=np.uint32
            The *k-mer* matches.
            Each row contains one match. Each match has the following
            columns:

                0. The sequence position in the input sequence
                1. The reference ID of the matched sequence in the table
                2. The sequence position of the matched sequence in the
                   table

        Notes
        -----

        The matches are ordered by the first column.

        Examples
        --------

        >>> sequence1 = ProteinSequence("BIQTITE")
        >>> table = BucketKmerTable.from_sequences(3, [sequence1], ref_ids=[100])
        >>> print(table)
        IQT: (100, 1)
        ITE: (100, 4)
        QTI: (100, 2)
        TIT: (100, 3)
        BIQ: (100, 0)
        >>> sequence2 = ProteinSequence("TITANITE")
        >>> print(table.match(sequence2))
        [[  0 100   3]
         [  5 100   4]]
        """
        cdef int INIT_SIZE = 1

        cdef int64 bucket
        cdef int64 self_kmer, other_kmer, sim_kmer
        cdef int64 match_i
        cdef int64 i, l
        cdef int64 length
        cdef uint32* bucket_ptr
        cdef uint32* array_stop

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
        cdef uint8[:] kmer_mask = _prepare_mask(
            self._kmer_alph, ignore_mask, len(sequence.code)
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
                    other_kmer = kmers[i]
                    bucket = other_kmer % self._n_buckets
                    bucket_ptr = <uint32*>ptr_array[bucket]
                    if bucket_ptr != NULL:
                        # There is at least one entry in this bucket
                        length = (<int64*>bucket_ptr)[0]
                        array_stop = bucket_ptr + length
                        bucket_ptr += 2
                        while bucket_ptr < array_stop:
                            self_kmer = (<int64*>bucket_ptr)[0]
                            if self_kmer == other_kmer:
                                # The k-mers are not only in the same
                                # bucket, but they are actually equal
                                if match_i >= matches.shape[0]:
                                    # The 'matches' array is full
                                    # -> double its size
                                    matches = expand(np.asarray(matches))
                                matches[match_i, 0] = i
                                bucket_ptr += 2
                                matches[match_i, 1] = bucket_ptr[0]
                                bucket_ptr += 1
                                matches[match_i, 2] = bucket_ptr[0]
                                bucket_ptr += 1
                                match_i += 1
                            else:
                                bucket_ptr += EntrySize.BUCKETS

        else:
            for i in range(kmers.shape[0]):
                if kmer_mask[i]:
                    other_kmer = kmers[i]
                    # If a similarity rule exists, iterate not only over
                    # the exact k-mer, but over all k-mers similar to
                    # the current k-mer
                    similar_kmers = similarity_rule.similar_kmers(
                        self._kmer_alph, other_kmer
                    )
                    for l in range(similar_kmers.shape[0]):
                        sim_kmer = similar_kmers[l]
                        bucket = sim_kmer % self._n_buckets
                        # Actual copy of the code from the other
                        # if-branch:
                        # It cannot be put properly in a cdef-function,
                        # as every function call would perform reference
                        # count changes and would decrease performance
                        bucket_ptr = <uint32*>ptr_array[bucket]
                        if bucket_ptr != NULL:
                            # There is at least one entry in this bucket
                            length = (<int64*>bucket_ptr)[0]
                            array_stop = bucket_ptr + length
                            bucket_ptr += 2
                            while bucket_ptr < array_stop:
                                self_kmer = (<int64*>bucket_ptr)[0]
                                if self_kmer == sim_kmer:
                                    # The k-mers are not only in the same
                                    # bucket, but they are actually equal
                                    if match_i >= matches.shape[0]:
                                        # The 'matches' array is full
                                        # -> double its size
                                        matches = expand(np.asarray(matches))
                                    matches[match_i, 0] = i
                                    bucket_ptr += 2
                                    matches[match_i, 1] = bucket_ptr[0]
                                    bucket_ptr += 1
                                    matches[match_i, 2] = bucket_ptr[0]
                                    bucket_ptr += 1
                                    match_i += 1
                                else:
                                    bucket_ptr += EntrySize.BUCKETS

        # Trim to correct size and return
        return np.asarray(matches[:match_i])


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def match_kmer_selection(self, positions, kmers):
        """
        match_kmer_selection(positions, kmers)

        Find matches between the *k-mers* in this table with the given
        *k-mer* selection.

        It is intended to use this method to find matches in a table
        that was created using :meth:`from_kmer_selection()`.

        Parameters
        ----------
        positions : ndarray, shape=(n,), dtype=uint32
            Sequence positions of the filtered subset of *k-mers* given
            in `kmers`.
        kmers : ndarray, shape=(n,), dtype=np.int64
            Filtered subset of *k-mer* codes to match against.

        Returns
        -------
        matches : ndarray, shape=(n,3), dtype=np.uint32
            The *k-mer* matches.
            Each row contains one *k-mer* match.
            Each match has the following columns:

                0. The sequence position of the input *k-mer*, taken
                   from `positions`
                1. The reference ID of the matched sequence in the table
                2. The sequence position of the matched *k-mer* in the
                   table

        Examples
        --------

        Reduce the size of sequence data in the table using minimizers:

        >>> sequence1 = ProteinSequence("THIS*IS*A*SEQVENCE")
        >>> kmer_alph = KmerAlphabet(sequence1.alphabet, k=3)
        >>> minimizer = MinimizerSelector(kmer_alph, window=4)
        >>> minimizer_pos, minimizers = minimizer.select(sequence1)
        >>> kmer_table = BucketKmerTable.from_kmer_selection(
        ...     kmer_alph, [minimizer_pos], [minimizers]
        ... )

        Use the same :class:`MinimizerSelector` to select the minimizers
        from the query sequence and match them against the table.
        Although the amount of *k-mers* is reduced, matching is still
        guanrateed to work, if the two sequences share identity in the
        given window:

        >>> sequence2 = ProteinSequence("ANQTHER*SEQVENCE")
        >>> minimizer_pos, minimizers = minimizer.select(sequence2)
        >>> matches = kmer_table.match_kmer_selection(minimizer_pos, minimizers)
        >>> print(matches)
        [[ 9  0 11]
         [12  0 14]]
        >>> for query_pos, _, db_pos in matches:
        ...     print(sequence1)
        ...     print(" " * (db_pos-1) + "^" * kmer_table.k)
        ...     print(sequence2)
        ...     print(" " * (query_pos-1) + "^" * kmer_table.k)
        ...     print()
        THIS*IS*A*SEQVENCE
          ^^^
        ANQTHER*SEQVENCE
                ^^^
        <BLANKLINE>
        THIS*IS*A*SEQVENCE
                    ^^^
        ANQTHER*SEQVENCE
                ^^^
        <BLANKLINE>
        """
        cdef int INIT_SIZE = 1

        cdef int64 i

        cdef int64 bucket
        cdef int64 self_kmer, other_kmer
        cdef int64 match_i
        cdef int64 seq_pos
        cdef int64 length
        cdef uint32* bucket_ptr
        cdef uint32* array_stop

        # Store in new variable
        # to disable repetitive initialization checks
        cdef ptr[:] ptr_array = self._ptr_array

        _check_kmer_bounds(kmers, self._kmer_alph)
        if positions.shape[0] != kmers.shape[0]:
            raise IndexError(
                f"{positions.shape[0]} positions were given "
                f"for {kmers.shape[0]} k-mers"
            )

        cdef uint32[:] pos_array = positions.astype(np.uint32, copy=False)
        cdef int64[:] kmer_array = kmers.astype(np.int64, copy=False)

        # This array will store the match positions
        # As the final number of matches is unknown, a list-like
        # approach is used:
        # The array is initialized with a relatively small inital size
        # and every time the limit would be exceeded its size is doubled
        cdef int64[:,:] matches = np.empty((INIT_SIZE, 3), dtype=np.int64)
        match_i = 0
        for i in range(kmer_array.shape[0]):
            other_kmer = kmer_array[i]
            seq_pos = pos_array[i]
            bucket = other_kmer % self._n_buckets
            bucket_ptr = <uint32*>ptr_array[bucket]
            if bucket_ptr != NULL:
                # There is at least one entry in this bucket
                length = (<int64*>bucket_ptr)[0]
                array_stop = bucket_ptr + length
                bucket_ptr += 2
                while bucket_ptr < array_stop:
                    self_kmer = (<int64*>bucket_ptr)[0]
                    if self_kmer == other_kmer:
                        # The k-mers are not only in the same
                        # bucket, but they are actually equal
                        if match_i >= matches.shape[0]:
                            # The 'matches' array is full
                            # -> double its size
                            matches = expand(np.asarray(matches))
                        matches[match_i, 0] = seq_pos
                        bucket_ptr += 2
                        matches[match_i, 1] = bucket_ptr[0]
                        bucket_ptr += 1
                        matches[match_i, 2] = bucket_ptr[0]
                        bucket_ptr += 1
                        match_i += 1
                    else:
                        bucket_ptr += EntrySize.BUCKETS

        # Trim to correct size and return
        return np.asarray(matches[:match_i])


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def count(self, kmers):
        """
        count(kmers=None)

        Count the number of occurences for each *k-mer* in the table.

        Parameters
        ----------
        kmers : ndarray, dtype=np.int64, optional
            The count is returned for these *k-mer* codes.
            By default all *k-mers* are counted in ascending order, i.e.
            ``count_for_kmer = counts[kmer]``.

        Returns
        -------
        counts : ndarray, dtype=np.int64, optional
            The counts for each given *k-mer*.

        Notes
        -----
        As each bucket need to be inspected for the actual *k-mer*
        entries, this method requires far more computation time than its
        :class:`KmerTable` equivalent.

        Examples
        --------
        >>> table = BucketKmerTable.from_sequences(
        ...     k = 2,
        ...     sequences = [NucleotideSequence("TTATA"), NucleotideSequence("CTAG")],
        ...     ref_ids = [0, 1]
        ... )
        >>> print(table)
        AG: (1, 2)
        AT: (0, 2)
        CT: (1, 0)
        TA: (0, 1), (0, 3), (1, 1)
        TT: (0, 0)

        Count two selected *k-mers*:

        >>> print(table.count(table.kmer_alphabet.encode_multiple(["TA", "AG"])))
        [3 1]
        """
        cdef int64 i

        cdef int64 bucket
        cdef int64 kmer, self_kmer
        cdef int64 length
        cdef uint32* bucket_ptr
        cdef uint32* array_stop
        cdef ptr[:] ptr_array = self._ptr_array

        _check_kmer_bounds(kmers, self._kmer_alph)
        cdef int64[:] kmer_array = kmers.astype(np.int64, copy=False)
        cdef int64[:] counts = np.zeros(kmer_array.shape[0], dtype=np.int64)

        for i in range(kmer_array.shape[0]):
            kmer = kmer_array[i]
            bucket = kmer % self._n_buckets
            bucket_ptr = <uint32*> (ptr_array[bucket])
            if bucket_ptr != NULL:
                length = (<int64*>bucket_ptr)[0]
                array_stop = bucket_ptr + length
                bucket_ptr += 2
                while bucket_ptr < array_stop:
                    self_kmer = (<int64*>bucket_ptr)[0]
                    if self_kmer == kmer:
                        counts[i] += 1
                    bucket_ptr += EntrySize.BUCKETS

        return np.asarray(counts)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_kmers(self):
        """
        Get the *k-mer* codes for all *k-mers* that have at least one
        position in the table.

        Returns
        -------
        kmers : ndarray, shape=(n,), dtype=np.int64
            The *k-mer* codes.

        Notes
        -----
        As each bucket need to be inspected for the actual *k-mer*
        entries, this method requires far more computation time than its
        :class:`KmerTable` equivalent.

        Examples
        --------

        >>> sequence = ProteinSequence("BIQTITE")
        >>> table = BucketKmerTable.from_sequences(3, [sequence], ref_ids=[100])
        >>> print(table)
        IQT: (100, 1)
        ITE: (100, 4)
        QTI: (100, 2)
        TIT: (100, 3)
        BIQ: (100, 0)
        >>> kmer_codes = table.get_kmers()
        >>> print(kmer_codes)
        [ 4360  4419  7879  9400 11701]
        >>> for code in kmer_codes:
        ...     print(table[code])
        [[100   1]]
        [[100   4]]
        [[100   2]]
        [[100   3]]
        [[100   0]]
        """
        cdef int64 bucket
        cdef int64 kmer
        cdef int64 length
        cdef uint32* bucket_ptr
        cdef uint32* array_stop
        cdef ptr[:] ptr_array = self._ptr_array

        cdef cpp_set[int64] kmer_set

        for bucket in range(ptr_array.shape[0]):
            bucket_ptr = <uint32*> (ptr_array[bucket])
            if bucket_ptr != NULL:
                length = (<int64*>bucket_ptr)[0]
                array_stop = bucket_ptr + length
                bucket_ptr += 2
                while bucket_ptr < array_stop:
                    kmer = (<int64*>bucket_ptr)[0]
                    kmer_set.insert(kmer)
                    bucket_ptr += EntrySize.BUCKETS

        cdef int64[:] kmers = np.zeros(kmer_set.size(), dtype=np.int64)
        cdef int64 i = 0
        for kmer in kmer_set:
            kmers[i] = kmer
            i += 1
        return np.sort(np.asarray(kmers))


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __getitem__(self, int64 kmer):
        cdef int64 i, j
        cdef int64 self_kmer
        cdef int64 length
        cdef uint32* bucket_ptr
        cdef uint32[:,:] positions

        if kmer >= len(self):
            raise AlphabetError(
                f"k-mer code {kmer} is out of bounds "
                f"for the given KmerAlphabet"
            )

        bucket_ptr = <uint32*>self._ptr_array[kmer % self._n_buckets]
        if bucket_ptr == NULL:
            return np.zeros((0, 2), dtype=np.uint32)
        else:
            length = (<int64*>bucket_ptr)[0]
            # Pessimistic array allocation:
            # All k-mer positions in bucket belong to the requested k-mer
            positions = np.empty(((length - 2) // 4, 2), dtype=np.uint32)
            i = 0
            for j in range(2, length, 4):
                self_kmer = bucket_ptr[j]
                if self_kmer == kmer:
                    positions[i,0] = bucket_ptr[j+2]
                    positions[i,1] = bucket_ptr[j+3]
                    i += 1
            # Trim to correct size
            return np.asarray(positions)[:i]


    def __len__(self):
        return len(self._kmer_alph)


    def __eq__(self, item):
        if item is self:
            return True
        if type(item) != BucketKmerTable:
            return False

        # Introduce static typing to access statically typed fields
        cdef BucketKmerTable other = item
        if self._kmer_alph.base_alphabet != other._kmer_alph.base_alphabet:
            return False
        if self._k != other._k:
            return False
        if self._n_buckets != other._n_buckets:
            return False
        return _equal_c_arrays(self._ptr_array, other._ptr_array)


    def __str__(self):
        return _to_string(self)


    def __getnewargs_ex__(self):
        return (self._n_buckets, self._kmer_alph), {}


    def __getstate__(self):
        return _pickle_c_arrays(self._ptr_array)

    def __setstate__(self, state):
        _unpickle_c_arrays(self._ptr_array, state)


    def __dealloc__(self):
        if self._is_initialized():
            _deallocate_ptrs(self._ptr_array)


    ## These private methods work analogous to KmerTable

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _count_kmers(self, int64[:] kmers):
        cdef uint32 seq_pos
        cdef int64 kmer

        cdef ptr[:] count_array = self._ptr_array

        for seq_pos in range(kmers.shape[0]):
            kmer = kmers[seq_pos]
            # Pool all k-mers that should go into the same bucket
            count_array[kmer % self._n_buckets] += 1

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _count_masked_kmers(self, int64[:] kmers, uint8[:] mask):
        cdef uint32 seq_pos
        cdef int64 kmer

        cdef ptr[:] count_array = self._ptr_array

        for seq_pos in range(kmers.shape[0]):
            if mask[seq_pos]:
                kmer = kmers[seq_pos]
                # Pool all k-mers that should go into the same bucket
                count_array[kmer % self._n_buckets] += 1


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_kmers(self, int64[:] kmers, uint32 ref_id, uint8[:] mask):
        cdef uint32 seq_pos
        cdef int64 current_size
        cdef int64 kmer
        cdef uint32* bucket_ptr
        cdef uint32* kmer_val_ptr

        # Store in new variable
        # to disable repetitive initialization checks
        cdef ptr[:] ptr_array = self._ptr_array

        if mask.shape[0] != kmers.shape[0]:
            raise IndexError(
                f"Mask has length {mask.shape[0]}, "
                f"but there are {kmers.shape[0]} k-mers"
            )

        for seq_pos in range(kmers.shape[0]):
            if mask[seq_pos]:
                kmer = kmers[seq_pos]
                bucket_ptr = <uint32*> ptr_array[kmer % self._n_buckets]

                # Append k-mer, reference ID and position
                current_size = (<int64*> bucket_ptr)[0]
                kmer_val_ptr = &bucket_ptr[current_size]
                (<int64*> kmer_val_ptr)[0] = kmer
                bucket_ptr[current_size + 2] = ref_id
                bucket_ptr[current_size + 3] = seq_pos
                (<int64*> bucket_ptr)[0] = current_size + EntrySize.BUCKETS


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_kmer_selection(self, uint32[:] positions, int64[:] kmers,
                         uint32 ref_id):
        cdef uint32 i
        cdef uint32 seq_pos
        cdef int64 current_size
        cdef int64 kmer
        cdef uint32* bucket_ptr
        cdef uint32* kmer_val_ptr

        if positions.shape[0] != kmers.shape[0]:
            raise IndexError(
                f"{positions.shape[0]} positions were given "
                f"for {kmers.shape[0]} k-mers"
            )

        # Store in new variable
        # to disable repetitive initialization checks
        cdef ptr[:] ptr_array = self._ptr_array

        for i in range(positions.shape[0]):
            kmer = kmers[i]
            seq_pos = positions[i]
            bucket_ptr = <uint32*> ptr_array[kmer % self._n_buckets]

            # Append k-mer reference ID and position
            current_size = (<int64*> bucket_ptr)[0]
            kmer_val_ptr = &bucket_ptr[current_size]
            (<int64*> kmer_val_ptr)[0] = kmer
            bucket_ptr[current_size + 2] = ref_id
            bucket_ptr[current_size + 3] = seq_pos
            (<int64*> bucket_ptr)[0] = current_size + EntrySize.BUCKETS


    cdef inline bint _is_initialized(self):
        try:
            if self._ptr_array is not None:
                return True
            else:
                return False
        except AttributeError:
            return False




@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _count_table_entries(ptr[:] count_array, ptr[:] ptr_array,
                         int64 element_size):
    """
    For each bucket, count the number of elements in `ptr_array` and add
    the count to the counts in `count_array`.
    The element size gives the number of 32 bit elements per entry.
    """
    cdef int64 length
    cdef int64 count
    cdef int64 bucket
    cdef uint32* bucket_ptr

    for bucket in range(count_array.shape[0]):
        bucket_ptr = <uint32*> (ptr_array[bucket])
        if bucket_ptr != NULL:
            # First 64 bits are length of C-array
            length = (<int64*>bucket_ptr)[0]
            count = (length - 2) // element_size
            count_array[bucket] += count


@cython.boundscheck(False)
@cython.wraparound(False)
def _init_c_arrays(ptr[:] ptr_array, int64 element_size):
    """
    Transform an array of counts into a pointer array, by replacing the
    count in each element with a pointer to an initialized but empty
    ``int32`` C-array.
    The size of each C-array is the count mutliplied by the
    `element_size`.
    The first element of each C-array is is the currently filled size
    of the C-array (an ``int64``) measured in number of ``int32``
    elements.
    """
    cdef int64 bucket
    cdef int64 count
    cdef uint32* bucket_ptr

    for bucket in range(ptr_array.shape[0]):
        # Before the C-array for a bucket initialized, the element in
        # the pointer array contains the number of elements the C-array
        # should hold
        count = ptr_array[bucket]
        if count != 0:
            # Array size + n x element size
            bucket_ptr = <uint32*>malloc(
                (2 + count * element_size) * sizeof(uint32)
            )
            if not bucket_ptr:
                raise MemoryError()
            # The initial size is 2,
            # which is the size of the array size value (int64)
            (<int64*> bucket_ptr)[0] = 2
            ptr_array[bucket] = <ptr>bucket_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
def _equal_c_arrays(ptr[:] self_ptr_array, ptr[:] other_ptr_array):
    """
    Check if two pointer arrays are equal, i.e. they point to C-arrays
    with equal elements.
    """
    cdef int64 bucket
    cdef int64 i
    cdef int64 self_length, other_length
    cdef uint32* self_bucket_ptr
    cdef uint32* other_bucket_ptr

    for bucket in range(self_ptr_array.shape[0]):
        self_bucket_ptr = <uint32*>self_ptr_array[bucket]
        other_bucket_ptr = <uint32*>other_ptr_array[bucket]
        if self_bucket_ptr != NULL or other_bucket_ptr != NULL:
            if self_bucket_ptr == NULL or other_bucket_ptr == NULL:
                # One of the tables has entries for this bucket
                # while the other one has not
                return False
            # This bucket exists in both tables
            self_length  = (<int64*>self_bucket_ptr )[0]
            other_length = (<int64*>other_bucket_ptr)[0]
            if self_length != other_length:
                return False
            for i in range(2, self_length):
                if self_bucket_ptr[i] != other_bucket_ptr[i]:
                    return False

    # If none of the previous checks failed, both objects are equal
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
def _append_entries(ptr[:] trg_ptr_array, ptr[:] src_ptr_array):
    """
    Append the elements in all C-arrays of the source pointer array to
    the corresponding C-arrays of the target pointer array.

    Expect that the target C-arrays are already initialized to
    sufficient capacity.
    """
    cdef int64 bucket
    cdef int64 self_length, other_length, new_length
    cdef uint32* self_kmer_ptr
    cdef uint32* other_kmer_ptr

    for bucket in range(trg_ptr_array.shape[0]):
        self_kmer_ptr = <uint32*>trg_ptr_array[bucket]
        other_kmer_ptr = <uint32*>src_ptr_array[bucket]
        if other_kmer_ptr != NULL:
            self_length  = (<int64*>self_kmer_ptr)[0]
            other_length = (<int64*>other_kmer_ptr)[0]
            # New new C-array needs the combucketed space of both
            # arrays, but only one length value
            new_length = self_length + other_length - 2
            (<int64*>self_kmer_ptr)[0] = new_length

            # Append the entry from the other table
            # to the entry in this table
            self_kmer_ptr += self_length
            other_kmer_ptr += 2
            memcpy(
                self_kmer_ptr, other_kmer_ptr,
                (other_length - 2) * sizeof(uint32)
                )


@cython.boundscheck(False)
@cython.wraparound(False)
def _pickle_c_arrays(ptr[:] ptr_array):
    """
    Pickle the C arrays into a single concatenated :class:`ndarray`.
    The lengths of each C-array on these concatenated array is saved as well.
    """
    cdef int64 pointer_i, bucket_i, concat_i
    cdef int64 length
    cdef uint32* bucket_ptr

    # First pass: Count the total concatenated size
    cdef int64 total_length = 0
    for pointer_i in range(ptr_array.shape[0]):
        bucket_ptr = <uint32*>ptr_array[pointer_i]
        if bucket_ptr != NULL:
            # The first element of the C-array is the length
            # of the array
            total_length += (<int64*>bucket_ptr)[0]

    # Second pass: Copy the C-arrays into a single concatenated array
    # and track the start position of each C-array
    cdef uint32[:] concatenated_array = np.empty(total_length, dtype=np.uint32)
    cdef int64[:] lengths = np.empty(ptr_array.shape[0], dtype=np.int64)
    concat_i = 0
    for pointer_i in range(ptr_array.shape[0]):
        bucket_ptr = <uint32*>ptr_array[pointer_i]
        if bucket_ptr != NULL:
            length = (<int64*>bucket_ptr)[0]
            lengths[pointer_i] = length
            memcpy(
                &concatenated_array[concat_i],
                bucket_ptr,
                length * sizeof(uint32),
            )
            concat_i += length
        else:
            lengths[pointer_i] = 0

    return np.asarray(concatenated_array), np.asarray(lengths)


@cython.boundscheck(False)
@cython.wraparound(False)
def _unpickle_c_arrays(ptr[:] ptr_array, state):
    """
    Unpickle the pickled `state` into the given `ptr_array`.
    """
    cdef int64 pointer_i, concat_i
    cdef int64 length
    cdef uint32* bucket_ptr

    cdef uint32[:] concatenated_array = state[0]
    cdef int64[:] lengths = state[1]

    concat_i = 0
    for pointer_i in range(ptr_array.shape[0]):
        length = lengths[pointer_i]
        if length != 0:
            bucket_ptr = <uint32*>malloc(length * sizeof(uint32))
            if not bucket_ptr:
                raise MemoryError
            memcpy(
                bucket_ptr,
                &concatenated_array[concat_i],
                length * sizeof(uint32),
            )
            concat_i += length
            ptr_array[pointer_i] = <ptr>bucket_ptr


cdef inline void _deallocate_ptrs(ptr[:] ptrs):
    cdef int64 kmer
    for kmer in range(ptrs.shape[0]):
        free(<uint32*>ptrs[kmer])


cdef np.ndarray expand(np.ndarray array):
    """
    Double the size of the first dimension of an existing 2D array.
    """
    new_array = np.empty(
        (array.shape[0] * 2, array.shape[1]), dtype=array.dtype
    )
    new_array[:array.shape[0], :] = array
    return new_array


def _prepare_mask(kmer_alphabet, ignore_mask, seq_length):
    """
    Convert an ignore mask into a positive mask.
    Multiple formats (boolean mask, pointer array, None) are supported
    for the input.
    """
    if ignore_mask is None:
        kmer_mask = np.ones(
            kmer_alphabet.kmer_array_length(seq_length), dtype=np.uint8
        )
    else:
        if not isinstance(ignore_mask, np.ndarray):
            raise TypeError(
                f"The given mask is a '{type(ignore_mask).__name__}', "
                f"but an ndarray was expected"
            )
        if ignore_mask.dtype != np.dtype(bool):
            raise ValueError("Expected a boolean mask")
        if len(ignore_mask) != seq_length:
            raise IndexError(
                f"ignore mask has length {len(ignore_mask)}, "
                f"but the length of the sequence is {seq_length}"
            )
        kmer_mask = _to_kmer_mask(
            np.frombuffer(
                ignore_mask.astype(bool, copy=False), dtype=np.uint8
            ),
            kmer_alphabet
        )
    return kmer_mask


@cython.boundscheck(False)
@cython.wraparound(False)
def _to_kmer_mask(uint8[:] mask not None, kmer_alphabet):
    """
    Transform a sequence ignore mask into a *k-mer* mask.

    The difference between those masks is that

        1. the *k-mer* mask is shorter and
        2. a position *i* in the *k-mer* mask is false, if any
           informative position of *k-mer[i]* is true in the ignore
           mask.
    """
    cdef int64 i, j
    cdef bint is_retained

    cdef uint8[:] kmer_mask = np.empty(
        kmer_alphabet.kmer_array_length(mask.shape[0]), dtype=np.uint8
    )
    cdef int64 offset
    cdef int64 k = kmer_alphabet.k
    cdef int64[:] spacing

    if kmer_alphabet.spacing is None:
        # Continuous k-mers
        for i in range(kmer_mask.shape[0]):
            is_retained = True
            # If any sequence position of this k-mer is removed,
            # discard this k-mer position
            for j in range(i, i + k):
                if mask[j]:
                    is_retained = False
            kmer_mask[i] = is_retained

    else:
        # Spaced k-mers
        spacing = kmer_alphabet.spacing
        for i in range(kmer_mask.shape[0]):
            is_retained = True
            # If any sequence position of this k-mer is removed,
            # discard this k-mer position
            for j in range(spacing.shape[0]):
                offset = spacing[j]
                if mask[j + offset]:
                    is_retained = False
            kmer_mask[i] = is_retained

    return np.asarray(kmer_mask)



def _check_position_shape(position_arrays, kmer_arrays):
    """
    Check if the given lists and each element have the same length
    and raise an exception, if this is not teh case.
    """
    if len(position_arrays) != len(kmer_arrays):
        raise IndexError(
            f"{len(position_arrays)} position arrays "
            f"for {len(kmer_arrays)} k-mer arrays were given"
        )
    for i, (positions, kmers) in enumerate(
        zip(position_arrays, kmer_arrays)
    ):
        if len(positions) != len(kmers):
            raise IndexError(
                f"{len(positions)} positions"
                f"for {len(kmers)} k-mers were given at index {i}"
            )


def _check_same_kmer_alphabet(tables):
    """
    Check if the *k-mer* alphabets of all tables are equal.
    """
    ref_alph = tables[0].kmer_alphabet
    for alph in (table.kmer_alphabet for table in tables):
        if not alph == ref_alph:
            raise ValueError(
                "The *k-mer* alphabets of the tables are not equal "
                "to each other"
            )


def _check_same_buckets(tables):
    """
    Check if the bucket sizes of all tables are equal.
    """
    ref_n_buckets = tables[0].n_buckets
    for buckets in (table.n_buckets for table in tables):
        if not buckets == ref_n_buckets:
            raise ValueError(
                "The number of buckets of the tables are not equal "
                "to each other"
            )


def _check_kmer_bounds(kmers, kmer_alphabet):
    """
    Check k-mer codes for out-of-bounds values.
    """
    if np.any(kmers < 0) or np.any(kmers >= len(kmer_alphabet)):
        raise AlphabetError(
            "Given k-mer codes do not represent valid k-mers"
        )


def _check_multiple_kmer_bounds(kmer_arrays, kmer_alphabet):
    """
    Check given arrays of k-mer codes for out-of-bounds values.
    """
    for kmers in kmer_arrays:
        if np.any(kmers < 0) or np.any(kmers >= len(kmer_alphabet)):
            raise AlphabetError(
                "Given k-mer codes do not represent valid k-mers"
            )


def _check_kmer_alphabet(kmer_alph):
    """
    Check if the given object is a KmerAaphabet and raise an exception,
    if this is not the case
    """
    if not isinstance(kmer_alph, KmerAlphabet):
        raise TypeError(
            f"Got {type(kmer_alph).__name__}, but KmerAlphabet was expected"
        )


def _compute_masks(masks, sequences):
    """
    Check, if the number of masks match the number of sequences, and
    raise an exception if this is not the case.
    If no masks are given, create a respective list of ``None`` values.
    """
    if masks is None:
        return [None] * len(sequences)
    else:
        if len(masks) != len(sequences):
            raise IndexError(
                f"{len(masks)} masks were given, "
                f"but there are {len(sequences)} sequences"
            )
        return masks


def _compute_ref_ids(ref_ids, sequences):
    """
    Check, if the number of reference IDs match the number of
    sequences, and raise an exception, if this is not the case.
    If no reference IDs are given, create an array that simply
    enumerates.
    """
    if ref_ids is None:
        return np.arange(len(sequences))
    else:
        if len(ref_ids) != len(sequences):
            raise IndexError(
                f"{len(ref_ids)} reference IDs were given, "
                f"but there are {len(sequences)} sequences"
            )
        return ref_ids


def _compute_alphabet(given_alphabet, sequence_alphabets):
    """
    If `given_alphabet` is None, find a common alphabet among
    `sequence_alphabets` and raise an exception if this is not possible.
    Otherwise just check compatibility of alphabets.
    """
    if given_alphabet is None:
        alphabet = common_alphabet(sequence_alphabets)
        if alphabet is None:
            raise ValueError(
                "There is no common alphabet that extends all alphabets"
            )
        return alphabet
    else:
        for alph in sequence_alphabets:
            if not given_alphabet.extends(alph):
                raise ValueError(
                    "The given alphabet is incompatible with a least one "
                    "alphabet of the given sequences"
                )
        return given_alphabet


def _to_string(table):
    lines = []
    for kmer in table.get_kmers():
        symbols = table.kmer_alphabet.decode(kmer)
        if isinstance(table.alphabet, LetterAlphabet):
            symbols = "".join(symbols)
        else:
            symbols = str(tuple(symbols))
        line = symbols + ": " + ", ".join(
            [str((ref_id.item(), pos.item())) for ref_id, pos in table[kmer]]
        )
        lines.append(line)
    return "\n".join(lines)
