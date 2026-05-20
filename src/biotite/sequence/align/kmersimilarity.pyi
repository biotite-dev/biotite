__all__ = ["SimilarityRule", "ScoreThresholdRule"]

import abc
from typing import Any
import numpy as np
from biotite.sequence.align.kmeralphabet import KmerAlphabet
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.typing import N, NDArray1

class SimilarityRule(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def similar_kmers(
        self, kmer_alphabet: KmerAlphabet[Any], kmer: int
    ) -> NDArray1[N, np.int64]: ...

class ScoreThresholdRule(SimilarityRule):
    def __init__(
        self, matrix: SubstitutionMatrix[Any, Any], threshold: int
    ) -> None: ...
    def similar_kmers(
        self, kmer_alphabet: KmerAlphabet[Any], kmer: int
    ) -> NDArray1[N, np.int64]: ...
