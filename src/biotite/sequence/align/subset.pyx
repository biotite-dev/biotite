# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["MinimizerRule", "SyncmerRule", "CachedSyncmerRule", "MincodeRule"]

cimport cython
cimport numpy as np

from numbers import Integral
import numpy as np
from .kmeralphabet import KmerAlphabet
from ..alphabet import AlphabetError


ctypedef np.int64_t int64
ctypedef np.uint32_t uint32


# Obtained from 'np.iinfo(np.int64).max'
DEF MAX_INT_64 = 9223372036854775807


class MinimizerRule:
    """
    MinimizerRule(kmer_alphabet, window, permutation=None)

    Find the *minimizers* from a given sequence.

    In a rolling window of *k-mers*, the minimizer is defined as the
    *k-mer* with the minimum *k-mer* code :footcite:`Roberts2004`.
    If the same minimum *k-mer* appears twice in a window, the leftmost
    *k-mer* is selected as minimizer.

    Parameters
    ----------
    kmer_alphabet : KmerAlphabet
        The *k-mer* alphabet that defines the *k-mer* size and the type
        of sequence this :class:`MinimizerRule` can be applied on.
    window : int
        The size of the rolling window, where the minimizers are
        searched in.
        In other words this is the number of *k-mers* per window.
        The window size must be at least 2.
    permutation : Permutation
        If set, the *k-mer* order is permuted, i.e.
        the minimizer is chosen based on the ordering of the sort keys
        from :class:`Permutation.permute()`.
        By default, the standard order of the :class:`KmerAlphabet` is
        used.
        This standard order is often the lexicographical order, which is
        known to yield suboptimal *density* in many cases
        :footcite:`Roberts2004`.
    
    Attributes
    ----------
    kmer_alphabet : KmerAlphabet
        The *k-mer* alphabet.
    window : int
        The window size.
    permutation : Permutation
        The permutation.

    Notes
    -----
    For minimizer computation a fast algorithm :footcite:`VanHerk1992`
    is used, whose runtime scales linearly with the length of the
    sequence and is constant with regard to the size of the rolling
    window.

    References
    ----------
    
    .. footbibliography::

    Examples
    --------

    The *k-mer* decomposition of a sequence can yield a high number of
    *k-mers*:

    >>> sequence1 = ProteinSequence("THIS*IS*A*SEQVENCE")
    >>> kmer_alph = KmerAlphabet(sequence1.alphabet, k=3)
    >>> all_kmers = kmer_alph.create_kmers(sequence1.code)
    >>> print(all_kmers)
    [ 9367  3639  4415  9199 13431  4415  9192 13271   567 13611  8725  2057
      7899  9875  1993  6363]
    >>> print(["".join(kmer_alph.decode(kmer)) for kmer in all_kmers])
    ['THI', 'HIS', 'IS*', 'S*I', '*IS', 'IS*', 'S*A', '*A*', 'A*S', '*SE', 'SEQ', 'EQV', 'QVE', 'VEN', 'ENC', 'NCE']

    Minimizers can be used to reduce the number of *k-mers* by selecting
    only the minimum *k-mer* in each window *w*:

    >>> minimizer = MinimizerRule(kmer_alph, window=4)
    >>> minimizer_pos, minimizers = minimizer.select(sequence1)
    >>> print(minimizer_pos)
    [ 1  2  5  8 11 14]
    >>> print(minimizers)
    [3639 4415 4415  567 2057 1993]
    >>> print(["".join(kmer_alph.decode(kmer)) for kmer in minimizers])
    ['HIS', 'IS*', 'IS*', 'A*S', 'EQV', 'ENC']

    Although this approach reduces the number of *k-mers*, minimizers
    are still guaranteed to match minimizers in another sequence, if
    they share an equal subsequence of at least length *w + k - 1*:

    >>> sequence2 = ProteinSequence("ANQTHER*SEQVENCE")
    >>> other_minimizer_pos, other_minimizers = minimizer.select(sequence2)
    >>> print(["".join(kmer_alph.decode(kmer)) for kmer in other_minimizers])
    ['ANQ', 'HER', 'ER*', 'EQV', 'ENC']
    >>> common_minimizers = set.intersection(set(minimizers), set(other_minimizers))
    >>> print(["".join(kmer_alph.decode(kmer)) for kmer in common_minimizers])
    ['EQV', 'ENC']
    """

    def __init__(self, kmer_alphabet, window, permutation=None):
        if window < 2:
            raise ValueError("Window size must be at least 2")
        self._window = window
        self._kmer_alph = kmer_alphabet
        self._permutation = permutation
    

    @property
    def kmer_alphabet(self):
        return self._kmer_alph
    
    @property
    def window(self):
        return self._window

    @property
    def permutation(self):
        return self._permutation
    

    def select(self, sequence, bint alphabet_check=True):
        """
        select(sequence, alphabet_check=True)

        Obtain all overlapping *k-mers* from a sequence and select
        the minimizers from them.

        Parameters
        ----------
        sequence : Sequence
            The sequence to find the minimizers in.
            Must be compatible with the given `kmer_alphabet`
        alphabet_check: bool, optional
            If set to false, the compatibility between the alphabets
            is not checked to gain additional performance.
        
        Returns
        -------
        minimizers_indices : ndarray, dtype=np.uint32
            The sequence indices where the minimizer *k-mers* start.
        minimizers : ndarray, dtype=np.int64
            The *k-mers* that are the selected minimizers, returned as
            *k-mer* code.
        
        Notes
        -----
        Duplicate minimizers are omitted, i.e. if two windows have the
        same minimizer position, the return values contain this
        minimizer only once.
        """
        if alphabet_check:
            if not self._kmer_alph.base_alphabet.extends(sequence.alphabet):
                raise ValueError(
                    "The sequence's alphabet does not fit the k-mer alphabet"
                )
        kmers = self._kmer_alph.create_kmers(sequence.code)
        return self.select_from_kmers(kmers)
    

    def select_from_kmers(self, kmers):
        """
        select_from_kmers(kmers)

        Select all overlapping *k-mers*.

        Parameters
        ----------
        kmers : ndarray, dtype=np.int64
            The *k-mer* codes representing the sequence to find the
            minimizers in.
            The *k-mer* codes correspond to the *k-mers* encoded by the
            given `kmer_alphabet`.
        alphabet_check: bool, optional
            If set to false, the compatibility between the alphabets
            is not checked to gain additional performance.
        
        Returns
        -------
        minimizers_indices : ndarray, dtype=np.uint32
            The indices in the input *k-mer* sequence where a minimizer
            appears.
        minimizers : ndarray, dtype=np.int64
            The corresponding *k-mers* codes of the minimizers.
        
        Notes
        -----
        Duplicate minimizers are omitted, i.e. if two windows have the
        same minimizer position, the return values contain this
        minimizer only once.
        """
        if self._permutation is None:
            ordering = kmers
        else:
            ordering = self._permutation.permute(kmers)
            if len(ordering) != len(kmers):
                raise IndexError(
                    f"The Permutation is defective, it gave {len(ordering)} "
                    f"sort keys for {len(kmers)} k-mers"
                )

        if len(kmers) < self._window:
            raise ValueError(
                "The number of k-mers is smaller than the window size"
            )
        return _minimize(
            kmers.astype(np.int64, copy=False),
            ordering.astype(np.int64, copy=False),
            self._window,
            include_duplicates=False
        )


class SyncmerRule:

    def __init__(self, alphabet, k, s, permutation=None, offset=(0,)):
        if not s < k:
            raise ValueError("s must be smaller than k")
        self._window = k - s + 1
        self._alphabet = alphabet
        self._kmer_alph = KmerAlphabet(alphabet, k)
        self._smer_alph = KmerAlphabet(alphabet, s)
        
        self._permutation = permutation

        self._offset = np.asarray(offset, dtype=np.int64)
        # Wrap around negative indices
        self._offset = np.where(
            self._offset < 0,
            self._window + self._offset,
            self._offset
        )
        if (self._offset >= self._window).any() or (self._offset < 0).any():
            raise IndexError(
                f"Offset is out of window range"
            )
        if len(np.unique(self._offset)) != len(self._offset):
            raise ValueError("Offset must contain unique values")
    

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def kmer_alphabet(self):
        return self._kmer_alph
    
    @property
    def smer_alphabet(self):
        return self._smer_alph

    @property
    def permutation(self):
        return self._permutation
    

    def select(self, sequence, bint alphabet_check=True):
        if alphabet_check:
            if not self._alphabet.extends(sequence.alphabet):
                raise ValueError(
                    "The sequence's alphabet does not fit the rule's alphabet"
                )
        kmers = self._kmer_alph.create_kmers(sequence.code)
        smers = self._smer_alph.create_kmers(sequence.code)

        if self._permutation is None:
            ordering = smers
        else:
            ordering = self._permutation.permute(smers)
            if len(ordering) != len(smers):
                raise IndexError(
                    f"The Permutation is defective, it gave {len(ordering)} "
                    f"sort keys for {len(smers)} s-mers"
                )

        # The aboslute position of the minimum s-mer for each k-mer
        min_pos, _ = _minimize(
            smers,
            ordering.astype(np.int64, copy=False),
            self._window,
            include_duplicates=True
        )
        # The position of the minimum s-mer relative to the start
        # of the k-mer
        relative_min_pos = min_pos - np.arange(len(kmers))
        syncmer_pos = self._filter_syncmer_pos(relative_min_pos)
        return syncmer_pos, kmers[syncmer_pos]
    

    def select_from_kmers(self, kmers):
        """
        The given `kmers` are not required to overlap.

        Notes
        -----
        Since for *s-mer* creation, the *k-mers* need to be converted
        back to symbol codes again and since the input *k-mers* are not
        required to overlap, calling :meth:`select()` is much faster.
        Howver, :meth:`select()` is only available, if a
        :class:`Sequence` object is available.
        """
        cdef int64 i
        
        symbol_codes_for_each_kmer = self._kmer_alph.split(kmers)
        
        cdef int64[:] min_pos = np.zeros(
            len(symbol_codes_for_each_kmer), dtype=np.int64
        )
        for i in range(symbol_codes_for_each_kmer.shape[0]):
            smers = self._smer_alph.create_kmers(symbol_codes_for_each_kmer[i])
            if self._permutation is None:
                ordering = smers
            else:
                ordering = self._permutation.permute(smers)
                if len(ordering) != len(smers):
                    raise IndexError(
                        f"The Permutation is defective, it gave {len(ordering)} "
                        f"sort keys for {len(smers)} s-mers"
                    )
            min_pos[i] = np.argmin(ordering)
        
        syncmer_pos = self._filter_syncmer_pos(min_pos)
        return syncmer_pos, kmers[syncmer_pos]
    

    def _filter_syncmer_pos(self, min_pos):
        """
        Get indices of *k-mers* that are syncmers, based on `min_pos`,
        the position of the minimum *s-mer* in each *k-mer*.
        Syncmers are k-mers whose the minimum s-mer is at (one of)
        the given offet position(s).
        """
        syncmer_mask = None
        for offset in self._offset:
            # For the usual number of offsets, this 'loop'-appoach is
            # faster than np.isin()
            if syncmer_mask is None:
                syncmer_mask = min_pos == offset
            else:
                syncmer_mask |= min_pos == offset
        return np.where(syncmer_mask)[0]


class CachedSyncmerRule(SyncmerRule):
    
    def __init__(self, alphabet, k, s, permutation=None, offset=(0,)):
        super().__init__(alphabet, k, s, permutation, offset)
        # Check for all possible *k-mers*, whether they are syncmers
        all_kmers = np.arange(len(self.kmer_alphabet))
        syncmer_indices, _ = super().select_from_kmers(all_kmers)
        # Convert the index array into a boolean mask
        self._syncmer_mask = np.zeros(len(self.kmer_alphabet), dtype=bool)
        self._syncmer_mask[syncmer_indices] = True
    

    def select(self, sequence, bint alphabet_check=True):
        if alphabet_check:
            if not self.alphabet.extends(sequence.alphabet):
                raise ValueError(
                    "The sequence's alphabet does not fit the rule's alphabet"
                )
        kmers = self.kmer_alphabet.create_kmers(sequence.code)
        return self.select_from_kmers(kmers)
    

    def select_from_kmers(self, kmers):
        """
        """
        syncmer_pos = np.where(self._syncmer_mask[kmers])[0]
        return syncmer_pos, kmers[syncmer_pos]


class MincodeRule:

    def __init__(self, kmer_alphabet, compression, permutation=None):
        if compression < 1:
            raise ValueError(
                "Compression factor must be equal to or larger than 1"
            )
        self._compression = compression
        self._kmer_alph = kmer_alphabet
        self._permutation = permutation
        if permutation is None:
            permutation_range = len(kmer_alphabet)
        else:
            permutation_range = permutation.max - permutation.min + 1
        self._threshold = permutation_range / compression
    

    @property
    def kmer_alphabet(self):
        return self._kmer_alph
    
    @property
    def compression(self):
        return self._compression

    @property
    def threshold(self):
        return self._threshold

    @property
    def permutation(self):
        return self._permutation
    

    def select(self, sequence, bint alphabet_check=True):
        if alphabet_check:
            if not self._kmer_alph.base_alphabet.extends(sequence.alphabet):
                raise ValueError(
                    "The sequence's alphabet does not fit the k-mer alphabet"
                )
        kmers = self._kmer_alph.create_kmers(sequence.code)
        return self.select_from_kmers(kmers)
    

    def select_from_kmers(self, kmers):
        if self._permutation is None:
            ordering = kmers
        else:
            ordering = self._permutation.permute(kmers)
            if len(ordering) != len(kmers):
                raise IndexError(
                    f"The Permutation is defective, it gave {len(ordering)} "
                    f"sort keys for {len(kmers)} k-mers"
                )

        mincode_pos = ordering <= self._threshold
        return mincode_pos, kmers[mincode_pos]
    

@cython.boundscheck(False)
@cython.wraparound(False)
def _minimize(int64[:] kmers, int64[:] ordering, uint32 window,
              bint include_duplicates):
    """
    Implementation of the algorithm originally devised by
    Marcel van Herk.

    In this implementation the frame is chosen differently:
    For a position 'x' the frame ranges from 'x' to 'x + window-1'
    instead of 'x - (window-1)/2' to 'x + (window-1)/2'.
    """
    cdef uint32 seq_i
    
    cdef uint32 n_windows = kmers.shape[0] - (window - 1)
    # Pessimistic array allocation size
    # -> Expect that every window has a new minimizer
    cdef uint32[:] mininizer_pos = np.empty(n_windows, dtype=np.uint32)
    cdef int64[:] minimizers = np.empty(n_windows, dtype=np.int64)
    # Counts the actual number of minimiers for later trimming
    cdef uint32 n_minimizers = 0

    # Variables for the position of the previous cumulative minimum
    # Assign an value that can never occur for the start,
    # as in the beginning there is no previous value
    cdef uint32 prev_argcummin = kmers.shape[0]
    # Variables for the position of the current cumulative minimum
    cdef uint32 combined_argcummin, forward_argcummin, reverse_argcummin
    # Variables for the current cumulative minimum
    cdef int64 combined_cummin, forward_cummin, reverse_cummin
    # Variables for cumulative minima at all positions
    cdef uint32[:] forward_argcummins = _chunk_wise_forward_argcummin(
        ordering, window
    )
    cdef uint32[:] reverse_argcummins = _chunk_wise_reverse_argcummin(
        ordering, window
    )

    for seq_i in range(n_windows):
        forward_argcummin = forward_argcummins[seq_i + window - 1]
        reverse_argcummin = reverse_argcummins[seq_i]
        forward_cummin = ordering[forward_argcummin]
        reverse_cummin = ordering[reverse_argcummin]
        
        # At ties the leftmost position is taken,
        # which stems from the reverse pass
        if forward_cummin < reverse_cummin:
            combined_argcummin = forward_argcummin
        else:
            combined_argcummin = reverse_argcummin
        
        # If the same minimizer position was observed before, the
        # duplicate is simply ignored, if 'include_duplicates' is false
        if include_duplicates or combined_argcummin != prev_argcummin:
            # Append minimizer to return value
            mininizer_pos[n_minimizers] = combined_argcummin
            minimizers[n_minimizers] = kmers[combined_argcummin]
            n_minimizers += 1
            prev_argcummin = combined_argcummin

    return (
        np.asarray(mininizer_pos)[:n_minimizers],
        np.asarray(minimizers)[:n_minimizers]
    )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef _chunk_wise_forward_argcummin(int64[:] values, uint32 chunk_size):
    """
    Argument of the cumulative minimum.
    """
    cdef uint32 seq_i

    cdef uint32 current_min_i = 0
    cdef int64 current_min, current_val
    cdef uint32[:] min_pos = np.empty(values.shape[0], dtype=np.uint32)
    
    # Any actual value will be smaller than this placeholder
    current_min = MAX_INT_64
    for seq_i in range(values.shape[0]):
        if seq_i % chunk_size == 0:
            # New chunk begins
            current_min = MAX_INT_64
        current_val = values[seq_i]
        if current_val < current_min:
            current_min_i = seq_i
            current_min = current_val
        min_pos[seq_i] = current_min_i
    
    return min_pos

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef _chunk_wise_reverse_argcummin(int64[:] values, uint32 chunk_size):
    """
    The same as above but starting from the other end and iterating
    backwards.
    Separation into two functions leads to code duplication.
    However, single implemention with reversed `values` as input
    has some disadvantages:

    - Indices must be transformed so that they point to the
      non-reversed `values`
    - There are issues in selecting the leftmost argument
    - An offset is necessary to ensure alignment of chunks with forward
      pass
    
    Hence, a separate 'reverse' variant of the function was implemented.
    """
    cdef uint32 seq_i

    cdef uint32 current_min_i = 0
    cdef int64 current_min, current_val
    cdef uint32[:] min_pos = np.empty(values.shape[0], dtype=np.uint32)
    
    current_min = MAX_INT_64
    for seq_i in reversed(range(values.shape[0])):
        # The chunk beginning is a small difference to forward
        # implementation, as it begins on the left of the chunk border
        if seq_i % chunk_size == chunk_size - 1:
            current_min = MAX_INT_64
        current_val = values[seq_i]
        # The '<=' is a small difference to forward implementation
        # to enure the loftmost argument is selected
        if current_val <= current_min:
            current_min_i = seq_i
            current_min = current_val
        min_pos[seq_i] = current_min_i
    
    return min_pos
