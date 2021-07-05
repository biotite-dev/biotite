# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["EValueEstimator"]

import numpy as np
from ..seqtypes import GeneralSequence
from .pairwise import align_optimal


class EValueEstimator:
    r"""
    This class is used to calculate *expect values* (E-values) for local
    pairwise sequence alignments.

    The E-value is a measure to quantify the significance of a found
    homology.
    It is the number of alignments, that would result from aligning
    random sequences of a given length, with a score at least as high as
    the score from an alignment of interest.

    The calculation of the E-value from score and sequence lengths
    depend on the two parameters :math:`\lambda` and :math:`K`
    :footcite:`Altschul1996`.
    These parameters are estimated from sampling a large number
    of random sequence alignments in :meth:`from_samples()`
    :footcite:`Altschul1986`, which may be time consuming.
    If these parameters are known, the constructor can be used instead.
    
    Based on the sampled parameters, the logarithm of the E-value can
    be quickly calculated via :meth:`log_evalue()`.

    Parameters
    ----------

    lam : float
        The :math:`\lambda` parameter.
    k : float
        The :math:`K` parameter.
    
    Notes
    -----

    The calculated E-value is a rough estimation that gets more
    accurate the more sequences are used in the sampling process.
    Note that the accuracy for alignment of short sequences, where the
    average length of a sampled alignment make up a significant part of
    the complete sampled sequence :footcite:`Altschul1996`.

    References
    ----------

    .. footbibliography

    Examples
    --------

    Create an alignment, whose significance should be evaluated.

    >>> query = NucleotideSequence("CGACGGCGTCTACGAGTCAACATCATTC")
    >>> hit = NucleotideSequence("GCTTTATTACGGGTTTACGAGTTCAACATCACGAAAACAA")
    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> gap_penalty = (-12, -2)
    >>> alignment = align_optimal(query, hit, matrix, gap_penalty, local=True)[0]
    >>> print(alignment)
    ACGGCGTCTACGAGT-CAACATCA
    ACGG-GTTTACGAGTTCAACATCA
    >>> print(alignment.score)
    77

    Create an estimator based on the same scoring scheme as the
    alignment.
    Use background symbol frequencies from the hypothetical reference
    database.

    >>> # Ensure deterministic results
    >>> np.random.seed(0)
    >>> # Sequences in database have a GC content of 0.6
    >>> background = np.array([0.2, 0.3, 0.3, 0.2])
    >>> estimator = EValueEstimator.from_samples(
    ...     query.alphabet, matrix, gap_penalty, background, sample_length=100
    >>> )

    Approach 1: Calculate E-value based on number of sequences in the
    hypothetical database (*100*).

    >>> log_e = estimator.log_evalue(alignment.score, len(query), 100 * len(hit))
    >>> print(f"E-value = {10**log_e:.2e}")
    E-value = 3.36e-01

    Approach 2: Calculate E-value based on total length of all sequences
    in the hypothetical database combined (*10000*).

    >>> log_e = estimator.log_evalue(alignment.score, len(query), 10000)
    >>> print(f"E-value = {10**log_e:.2e}")
    E-value = 8.41e-01
    """

    def __init__(self, lam, k):
        self._lam = lam
        self._k = k

    @staticmethod
    def from_samples(alphabet, matrix, gap_penalty, background_frequencies,
                     sample_length=1000, sample_size=1000):
        random_sequence_code = np.random.choice(
            len(alphabet),
            size=(sample_size, 2, sample_length),
            p=background_frequencies
        )

        sample_scores = np.zeros(sample_size, dtype=int)
        for i in range(sample_size):
            seq1 = GeneralSequence(alphabet)
            seq2 = GeneralSequence(alphabet)
            seq1.code = random_sequence_code[i,0]
            seq2.code = random_sequence_code[i,1]
            sample_scores[i] = align_optimal(
                seq1, seq2, matrix,
                local=True, gap_penalty=gap_penalty, max_number=1
            )[0].score
        
        lam = np.pi / np.sqrt(6 * np.var(sample_scores))
        u = np.mean(sample_scores) - np.euler_gamma / lam
        k = np.exp(lam * u) / sample_length**2
        
        return EValueEstimator(lam, k)

    @property
    def lam(self):
        return self._lam
    
    @property
    def k(self):
        return self._k
    
    def log_evalue(self, score, seq1_length, seq2_length):
        """
        Parameters
        ----------
        seq1_length : int
            In the context of an homology search in a sequence database,
            this is usually the length of the query sequence.
        """
        score = np.asarray(score)
        seq1_length = np.asarray(seq1_length)
        seq2_length = np.asarray(seq2_length)

        return np.log10(self._k * seq1_length * seq2_length) \
            - self._lam * score / np.log(10)