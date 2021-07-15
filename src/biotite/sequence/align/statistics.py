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
    
    Based on the sampled parameters, the decadic logarithm of the
    E-value can be quickly calculated via :meth:`log_evalue()`.

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

    .. footbibliography::

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
    ... )

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
    def from_samples(alphabet, matrix, gap_penalty, frequencies,
                     sample_length=1000, sample_size=1000):
        r"""
        Create an :class:`EValueEstimator` with :math:`\lambda` and
        :math:`K` estimated via sampling alignments of random sequences
        based on a given scoring scheme.

        The parameters are estimated from the sampled alignment scores
        using the method of moments :footcite:`Altschul1986`.

        Parameters
        ----------
        alphabet : Alphabet, length=k
            The alphabet for the sampled sequences.
        matrix : SubstitutionMatrix
            The substitution matrix.
            It must be compatible with the given `alphabet` and the
            expected similarity score between two random symbols must be
            negative.
        gap_penalty : int or tuple(int,int)
            Either a linear (``int``) or affine (``tuple``) gap penalty.
            Integers must be negative.
        frequencies : ndarray, shape=k, dtype=float
            The background frequencies for each symbol in the
            `alphabet`.
            The random sequences are created based on these frequencies.
        sample_length : int
            The length of the sampled sequences.
            It should be much larger than the average length of a local
            alignment of two sequences.
            The runtime scales quadratically with this parameter.
        sample_size : int
            The number of sampled sequences.
            The accuracy of the estimated parameters and E-values,
            but also the runtime increases with the sample size.
        
        Returns
        -------
        estimator : EValueEstimator
            A :class:`EValueEstimator` with sampled :math:`\lambda` and
            :math:`K` parameters.
        
        Notes
        -----
        The sampling process generates random sequences based on
        ``numpy.random``.
        To ensure reproducible results you could call
        :func:`numpy.random.seed()` before running
        :meth:`from_samples()`.
        """
        if len(frequencies) != len(alphabet):
            raise IndexError(
                f"Background frequencies for {len(frequencies)} symbols were "
                f"given, but the alphabet has {len(alphabet)} symbols"
            )
        if np.any(frequencies < 0):
            raise ValueError("Background frequencies must be positive")
        # Normalize background frequencies
        frequencies = frequencies / np.sum(frequencies)

        # Check matrix
        if not matrix.is_symmetric():
            raise ValueError("A symmetric substitution matrix is required")
        if not matrix.get_alphabet1().extends(alphabet):
            raise ValueError(
                "The substitution matrix is not compatible "
                "with the given alphabet"
            )
        score_matrix = matrix.score_matrix()[:len(alphabet), :len(alphabet)]
        if np.sum(
            score_matrix \
            * frequencies[np.newaxis, :] \
            * frequencies[:, np.newaxis]
        ) >= 0:
            raise ValueError(
                "Invalid substitution matrix, the expected similarity "
                "score between two random symbols is not negative"
            )

        # Generate the sequence code for the random sequences
        random_sequence_code = np.random.choice(
            len(alphabet),
            size=(sample_size, 2, sample_length),
            p=frequencies
        )

        # Sample the alignments of random sequences
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
        
        # Use method of moments to estimate parameters
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
        r"""
        Calculate the decadic logarithm of the E-value for a given
        score.

        The E-value and the logarithm of the E-value is calculated as

        .. math::
        
            E = Kmn e^{-\lambda s}

            \log_{10} E = (\log_{10} Kmn) - \frac{\lambda s}{\ln 10},
        
        where :math:`s` is the similarity score and :math:`m` and
        :math:`n` are the lengths of the aligned sequences.

        Parameters
        ----------
        score : int or ndarray, dtype=int
            The score to evaluate.
        seq1_length : int or ndarray, dtype=int
            The length of the first sequence.
            In the context of a homology search in a sequence database,
            this is usually the length of the query sequence.
        seq2_length : int or ndarray, dtype=int
            The length of the second sequence.
            In the context of a homology search in a sequence database,
            this is usually either the combined length of all sequences
            in the database or the length of the hit sequence multiplied
            by the number of sequences in the database.
        
        Returns
        -------
        log_e : float
            The decadic logarithm of the E-value.
        
        Notes
        -----
        This method returns the logarithm of the E-value instead of
        the E-value, as low E-values indicating a highly significant
        homology cannot be accurately represented by a ``float``.
        """
        score = np.asarray(score)
        seq1_length = np.asarray(seq1_length)
        seq2_length = np.asarray(seq2_length)

        return np.log10(self._k * seq1_length * seq2_length) \
            - self._lam * score / np.log(10)