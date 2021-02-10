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
        return np.arange(kmer_number(k, len(alphabet)))[:, np.newaxis]


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

    def __init__(self, similarities):
        self._sim = similarities

    @abc.abstractmethod
    def similarities(self, k, alphabet):
        n_kmers = kmer_number(k, len(alphabet))
        if len(self._sim) != n_kmers:
            raise ValueError(
                f"similarities has length {len(self._sim)}, "
                f"but there are {n_kmers} possible k-mers"
            )
        return self._sim