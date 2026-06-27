# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["SimilarityRule", "ScoreThresholdRule"]

import abc
from typing import Any, Generic
import numpy as np
from biotite.rust.sequence.align import similar_kmers as rust_similar_kmers
from biotite.sequence.align.kmeralphabet import KmerAlphabet
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.typing import N, NDArray1, S


class SimilarityRule(metaclass=abc.ABCMeta):
    """
    This is the abstract base class for all similarity rules.
    A :class:`SimilarityRule` calculates all *similar* *k-mers* for
    a given *k-mer*, while the definition of similarity depends
    on the derived class.
    """

    @abc.abstractmethod
    def similar_kmers(
        self, kmer_alphabet: KmerAlphabet[Any], kmer: int
    ) -> NDArray1[N, np.int64]:
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
        raise NotImplementedError


class ScoreThresholdRule(SimilarityRule, Generic[S]):
    """
    __init__(matrix, threshold)

    This similarity rule calculates all *k-mers* that have a greater or
    equal similarity score with a given *k-mer* than a defined threshold
    score.

    The similarity score :math:`S` of two *k-mers* :math:`a` and
    :math:`b` is defined as the sum of the pairwise similarity scores
    from a substitution matrix :math:`M`:

    .. math::

        S(a,b) = \\sum_{i=1}^k M(a_i, b_i)

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

    def __init__(self, matrix: SubstitutionMatrix[S, S], threshold: int) -> None:
        if not matrix.is_symmetric():
            raise ValueError("A symmetric substitution matrix is required")
        self._matrix = matrix
        self._threshold = int(threshold)
        # The contiguous score matrix and the per-symbol maximum scores only
        # depend on the matrix, so they are computed once here instead of on
        # every `similar_kmers()` call
        self._score_matrix = np.ascontiguousarray(matrix.score_matrix(), dtype=np.int32)
        self._max_scores = np.max(self._score_matrix, axis=-1).astype(
            np.int32, copy=False
        )

    def similar_kmers(
        self, kmer_alphabet: KmerAlphabet[S], kmer: int
    ) -> NDArray1[N, np.int64]:
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
        if not self._matrix.get_alphabet1().extends(kmer_alphabet.base_alphabet):
            raise ValueError(
                "Substitution matrix is incompatible with k-mer base alphabet"
            )

        # Split the k-mer code into the individual symbol codes
        split_kmer = kmer_alphabet.split(kmer).astype(np.int64)
        similar_split_kmers = rust_similar_kmers(
            self._score_matrix,
            self._max_scores,
            split_kmer,
            len(kmer_alphabet.base_alphabet),
            self._threshold,
        )
        # Convert the split k-mers back to k-mer codes
        return kmer_alphabet.fuse(similar_split_kmers)
