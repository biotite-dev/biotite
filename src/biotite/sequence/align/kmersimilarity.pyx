# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["SimilarityRule", "ScoreThresholdRule"]

cimport cython
cimport numpy as np

import abc
import numpy as np


ctypedef np.int64_t int64
ctypedef np.int32_t int32


class SimilarityRule(metaclass=abc.ABCMeta):
    """
    This is the abstract base class for all similarity rules.
    A :class:`SimilarityRule` calculates all *similar* *k-mers* for
    a given *k-mer*, while the definition of similarity depends
    on the derived class.
    """

    @abc.abstractmethod
    def similar_kmers(self, kmer_alphabet, kmer):
        """
        similar_kmers(kmer_alphabet, kmer)

        Calculate all similar *k-mers* for a given *k-mer*.

        Parameters
        ----------
        kmer_alphabet : KmerAlphabet
            The reference *k-mer* alphabet to select the *k-mers* from.
        kmer : int
            The symbol code for the *k-mer* to find similars for.
        
        Returns
        -------
        similar_kmers : ndarray, dtype=np.int64
            The symbol codes for all similar *k-mers*.
        
        Notes
        -----
        The implementations in derived classes must ensure that the
        returned array

            1. contains no duplicates and
            2. includes the input `kmer` itself.
        """
        pass


class ScoreThresholdRule(SimilarityRule):
    """
    __init__(matrix, threshold)

    This similarity rule calculates all *k-mers* that have a greater or
    equal similarity score with a given *k-mer* than a defined threshold 
    score.

    The similarity score :math:`S` of two *k-mers* :math:`a` and
    :math:`b` is defined as the sum of the pairwise similarity scores
    from a substitution matrix :math:`M`:

    .. math::

        S(a,b) = \sum_{i=1}^k M(a_i, b_i)

    Therefore, this similarity rule allows substitutions with similar
    symbols within a *k-mer*.

    This class is especially useful for finding similar *k-mers* in
    protein sequences.

    Parameters
    ----------
    matrix : SubstitutionMatrix
        The similarity scores are taken from this matrix.
        The matrix must be symmetric.
    threshold : int
        The threshold score.
        A *k-mer* :math:`b` is regarded as similar to a *k-mer*
        :math:`a`, if the similarity score between :math:`a` and
        :math:`b` is equal or greater than the threshold.
    
    Notes
    -----
    For efficient generation of similar *k-mers* an implementation of
    the *branch-and-bound* algorithm :footcite:`Hauser2013` is used.

    References
    ----------
    
    .. footbibliography::
    
    Examples
    --------

    >>> kmer_alphabet = KmerAlphabet(ProteinSequence.alphabet, k=3)
    >>> matrix = SubstitutionMatrix.std_protein_matrix()
    >>> rule = ScoreThresholdRule(matrix, threshold=15)
    >>> similars = rule.similar_kmers(kmer_alphabet, kmer_alphabet.encode("AIW"))
    >>> print(["".join(s) for s in kmer_alphabet.decode_multiple(similars)])
    ['AFW', 'AIW', 'ALW', 'AMW', 'AVW', 'CIW', 'GIW', 'SIW', 'SVW', 'TIW', 'VIW', 'XIW']
    """

    def __init__(self, matrix, int32 threshold):
        if not matrix.is_symmetric():
            raise ValueError("A symmetric substitution matrix is required")
        self._matrix = matrix
        self._threshold = threshold

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def similar_kmers(self, kmer_alphabet, kmer):
        """
        Calculate all similar *k-mers* for a given *k-mer*.

        Parameters
        ----------
        kmer_alphabet : KmerAlphabet
            The reference *k-mer* alphabet to select the *k-mers* from.
        kmer : int
            The symbol code for the *k-mer* to find similars for.
        
        Returns
        -------
        similar_kmers : ndarray, dtype=np.int64
            The symbol codes for all similar *k-mers*.
        """
        cdef int INIT_SIZE = 1
        
        if not self._matrix.get_alphabet1().extends(
            kmer_alphabet.base_alphabet
        ):
            raise ValueError(
                "Substitution matrix is incompatible with k-mer base alphabet"
            )

        cdef int64 alph_len = len(kmer_alphabet.base_alphabet)
        cdef const int32[:,:] matrix = self._matrix.score_matrix()
        # For simplicity trim matrix to required size
        # (remove unused symbols)
        matrix = matrix[:alph_len, :alph_len]
        cdef int32 threshold = self._threshold

        cdef int32[:] max_scores = np.max(self._matrix.score_matrix(), axis=-1)

        cdef int k = kmer_alphabet.k
        # Split the k-mer code into the individual symbol codes 
        cdef int64[:] split_kmer = kmer_alphabet.split(kmer).astype(np.int64)
        # This array will hold the current kmer to be tested
        cdef int64[:] current_split_kmer = np.zeros(k, dtype=np.int64)
        # This array will store the accepted k-mers
        # i.e. k-mers that reach the threshold score
        cdef int64[:,:] similar_split_kmers = np.zeros(
            (INIT_SIZE, k), dtype=np.int64
        )

        # Calculate the minimum score for each k-mer position that is
        # necessary to reach a total higher/equal to the threshold score
        cdef int32[:] positional_thresholds = np.empty(k, dtype=np.int32)
        cdef int i
        cdef int total_max_score = 0
        for i in reversed(range(positional_thresholds.shape[0])):
            positional_thresholds[i] = threshold - total_max_score
            total_max_score += max_scores[split_kmer[i]]

        # 'pos' is the current position within the k-mer
        # where symbols are substituted
        cdef int pos = 0
        cdef int similar_i = 0
        cdef int32 score
        # 'pos' is -1, after all symbol codes at pos 0 are traversed
        while pos != -1:
            if current_split_kmer[pos] >= alph_len:
                # All symbol codes were traversed at this position
                # -> jump one k-mer position back and proceed with
                # next symbol
                pos -= 1
                if pos != -1:
                    current_split_kmer[pos] += 1
            else:
                # Get total similarity score between the input k-mer
                # and generated k-mer up to the point of the current
                # position
                score = 0
                for i in range(pos+1):
                    score += matrix[split_kmer[i], current_split_kmer[i]]
                # Check score threshold condition
                if score >= positional_thresholds[pos]:
                    # Threshold condition is fulfilled:
                    # Either go deeper in the same branch
                    # (jump one position forward) ...
                    if pos < k-1:
                        pos += 1
                        current_split_kmer[pos] = 0
                    # ...or store similar k-mer,
                    # if already at maximum depth (last k-mer position)
                    else:
                        if similar_i >= similar_split_kmers.shape[0]:
                            # The array is full -> double its size
                            similar_split_kmers = expand(
                                np.asarray(similar_split_kmers)
                            )
                        similar_split_kmers[similar_i] = current_split_kmer
                        similar_i += 1
                        # Proceed with the next symbol at this position,
                        # as we cannot go deeper anymore
                        current_split_kmer[pos] += 1
                else:
                    # The threshold score is not reached
                    # -> this branch ends and we proceed with the next
                    # symbol at this position
                    current_split_kmer[pos] += 1
        
        # Trim to correct size
        # and convert split k-mers back to k-mer code
        return kmer_alphabet.fuse(np.asarray(similar_split_kmers[:similar_i]))


cdef np.ndarray expand(np.ndarray array):
    """
    Double the size of the first dimension of an existing array.
    """
    new_array = np.empty((array.shape[0]*2, array.shape[1]), dtype=array.dtype)
    new_array[:array.shape[0],:] = array
    return new_array