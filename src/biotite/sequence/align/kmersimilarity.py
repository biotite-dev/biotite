# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["SimilarityRule", "IdentityRule"]

import abc
import numpy as np
from .kmer import kmer_number


class SimilarityRule(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def similarities(self, k, alphabet):
        pass
    
    @property
    def k(self):
        return self._k


class IdentityRule(SimilarityRule):

    def similarities(self, k, alphabet):
        return np.identity(kmer_number(k, len(alphabet)), dtype=bool)


class ScoreThresholdRule(SimilarityRule):

    def __init__(self, matrix):
        self._matrix = matrix

    @abc.abstractmethod
    def similarities(self, k, alphabet):
        raise NotImplementedError()


class MatchNumberRule(SimilarityRule):

    def __init__(self, min_matches):
        self._matches = min_matches

    @abc.abstractmethod
    def similarities(self, k, alphabet):
        raise NotImplementedError()


class CustomRule(SimilarityRule):

    def __init__(self, mask):
        self._mask = mask

    @abc.abstractmethod
    def similarities(self, k, alphabet):
        n_kmers = kmer_number(k, len(alphabet))
        if self._mask.shape != (n_kmers, n_kmers):
            raise ValueError(
                f"Mask is an {self._mask.shape} matrix, "
                f"but there are {n_kmers} possible k-mers"
            )
        return self._mask