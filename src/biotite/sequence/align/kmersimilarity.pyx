# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

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
    Self-similar
    No duplicates
    """

    @abc.abstractmethod
    def similar_kmers(self, kmer_alphabet, kmer): 
        pass


class ScoreThresholdRule(SimilarityRule):

    def __init__(self, matrix, int32 threshold):
        if not matrix.is_symmetric():
            raise ValueError("A symmetric substitution matrix is required")
        self._matrix = matrix
        self._threshold = threshold

    def similar_kmers(self, kmer_alphabet, kmer):
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
        # This array will store the accpeted k-mers
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

        cdef int pos = 0
        cdef int similar_i = 0
        cdef int32 score
        # 'pos' is -1, after all symbol codes at pos 0 are traversed
        while pos != -1:
            if current_split_kmer[pos] >= alph_len:
                # All symbol codes were traversed at this position
                # -> jump one k-mer position back
                pos -= 1
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
                    # Either go deeper in the same branch...
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