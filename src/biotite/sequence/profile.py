# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import warnings
from numbers import Integral
import numpy as np
from biotite.sequence.align.alignment import get_codes
from biotite.sequence.alphabet import LetterAlphabet
from biotite.sequence.seqtypes import (
    GeneralSequence,
    NucleotideSequence,
    ProteinSequence,
)

__name__ = "biotite.sequence"
__author__ = "Maximilian Greil"
__all__ = ["SequenceProfile"]

# Abbreviations
_NUC_DNA_ALPH = NucleotideSequence.alphabet_unamb
_NUC_RNA_ALPH = LetterAlphabet(["A", "C", "G", "U"])
_PROT_ALPH = ProteinSequence.alphabet


def _determine_common_alphabet(alphabets):
    """
    Determine the common alphabet from a list of alphabets, that
    extends all alphabets.
    """
    common_alphabet = alphabets[0]
    for alphabet in alphabets[1:]:
        if not common_alphabet.extends(alphabet):
            if alphabet.extends(common_alphabet):
                common_alphabet = alphabet
            else:
                raise ValueError(
                    "There is no common alphabet that extends all alphabets"
                )
    return common_alphabet


def _codes_to_iupac(frequency, codes, maxes, row):
    """
    Returns IUPAC code for a row of 'symbols' with none, one or
    multiple maximum positions.
    """
    if np.sum(frequency) == 0:
        raise ValueError(
            f"There is an empty column in the 'symbols' frequency table. "
            f"This doesn't make sense in context of an alignment. "
            f"Please check the 'symbols' frequency table in row {row}."
        )
    key = tuple(np.where(frequency == maxes)[0])
    return codes[key]


class SequenceProfile(object):
    """
    A :class:`SequenceProfile` object stores information about a
    sequence profile of aligned sequences.
    It is possible to calculate and return its consensus sequence.

    This class saves the position frequency matrix
    (position count matrix) 'symbols' of the occurrences of each
    alphabet symbol at each position.
    It also saves the number of gaps at each position in the array
    'gaps'.

    With :meth:`from_alignment()` a :class:`SequenceProfile` object can
    be created from an indefinite number of aligned sequences.

    With :meth:`probability_matrix()` the position probability matrix
    can be created based on 'symbols' and a pseudocount.

    With :meth:`log_odds_matrix()` the position weight matrix can
    be created based on the before calculated position probability
    matrix and the background frequencies.

    With :meth:`sequence_probability_from_matrix()` the probability of a
    sequence can be calculated based on the before calculated position
    probability matrix of this instance of object SequenceProfile.

    With :meth:`sequence_score_from_matrix()` the score of a sequence
    can be calculated based on the before calculated position weight
    matrix of this instance of object SequenceProfile.

    All attributes of this class are publicly accessible.

    Parameters
    ----------
    symbols : ndarray, dtype=int, shape=(n,k)
        This matrix simply saves for each position how often absolutely
        each symbol is present.
    gaps : ndarray, dtype=int, shape=n
        Array which indicates the number of gaps at each position.
    alphabet : Alphabet, length=k
        Alphabet of sequences of sequence profile.

    Attributes
    ----------
    symbols : ndarray, dtype=int, shape=(n,k)
        This matrix simply saves for each position how often absolutely
        each symbol is present.
    gaps : ndarray, dtype=int, shape=n
        Array which indicates the number of gaps at each position.
    alphabet : Alphabet, length=k
        Alphabet of sequences of sequence profile

    Examples
    --------

    Create a profile from a multiple sequence alignment:

    >>> sequences = [
    ...     NucleotideSequence("CGCTCATTC"),
    ...     NucleotideSequence("CGCTATTC"),
    ...     NucleotideSequence("CCCTCAATC"),
    ... ]
    >>> msa, _, _, _ = align_multiple(
    ...     sequences, SubstitutionMatrix.std_nucleotide_matrix(), gap_penalty=-5
    ... )
    >>> print(msa)
    CGCTCATTC
    CGCT-ATTC
    CCCTCAATC
    >>> profile = SequenceProfile.from_alignment(msa)
    >>> print(profile)
      A C G T
    0 0 3 0 0
    1 0 1 2 0
    2 0 3 0 0
    3 0 0 0 3
    4 0 2 0 0
    5 3 0 0 0
    6 1 0 0 2
    7 0 0 0 3
    8 0 3 0 0
    >>> print(profile.gaps)
    [0 0 0 0 1 0 0 0 0]

    Slice the profile (masks and index arrays are also supported):

    >>> print(profile[2:])
      A C G T
    0 0 3 0 0
    1 0 0 0 3
    2 0 2 0 0
    3 3 0 0 0
    4 1 0 0 2
    5 0 0 0 3
    6 0 3 0 0

    Use the profile to compute the position probability matrix:

    >>> print(profile.probability_matrix())
    [[0.000 1.000 0.000 0.000]
     [0.000 0.333 0.667 0.000]
     [0.000 1.000 0.000 0.000]
     [0.000 0.000 0.000 1.000]
     [0.000 1.000 0.000 0.000]
     [1.000 0.000 0.000 0.000]
     [0.333 0.000 0.000 0.667]
     [0.000 0.000 0.000 1.000]
     [0.000 1.000 0.000 0.000]]
    """

    def __init__(self, symbols, gaps, alphabet):
        self._symbols = symbols
        self._gaps = gaps
        self._alphabet = alphabet

        if len(alphabet) != symbols.shape[1]:
            raise ValueError(
                f"The given alphabet doesn't have the same length "
                f"({len(alphabet)}) as the number of columns "
                f"({symbols.shape[1]}) in the 'symbols' frequency table."
            )

        if gaps.shape[0] != symbols.shape[0]:
            raise ValueError(
                f"The given 'gaps' position matrix doesn't have the same "
                f"length ({gaps.shape[0]}) as the 'symbols' "
                f"frequency table ({symbols.shape[0]})"
            )

    @property
    def symbols(self):
        return self._symbols

    @property
    def gaps(self):
        return self._gaps

    @property
    def alphabet(self):
        return self._alphabet

    @symbols.setter
    def symbols(self, new_symbols):
        if not new_symbols.shape == self.symbols.shape:
            raise ValueError(
                f"New ndarray 'symbols' must be of same shape "
                f"{self.symbols.shape} as the old one"
            )
        self._symbols = new_symbols

    @gaps.setter
    def gaps(self, new_gaps):
        if not new_gaps.shape == self.gaps.shape:
            raise ValueError(
                f"New ndarray 'gaps' must be of same shape "
                f"{self.gaps.shape} as the old one"
            )
        self._gaps = new_gaps

    def __str__(self):
        # Add an additional row and column for the position and symbol indicators
        print_matrix = np.full(
            (self.symbols.shape[0] + 1, self.symbols.shape[1] + 1), "", dtype=object
        )
        print_matrix[1:, 1:] = self.symbols.astype(str)
        print_matrix[0, 1:] = [str(sym) for sym in self.alphabet]
        print_matrix[1:, 0] = [str(i) for i in range(self.symbols.shape[0])]
        max_len = len(max(print_matrix.flatten(), key=len))
        return "\n".join(
            [
                " ".join([str(cell).rjust(max_len) for cell in row])
                for row in print_matrix
            ]
        )

    def __repr__(self):
        return (
            f"SequenceProfile(np.{np.array_repr(self.symbols)}, "
            f"np.{np.array_repr(self.gaps)}, Alphabet({self.alphabet}))"
        )

    def __eq__(self, item):
        if not isinstance(item, SequenceProfile):
            return False
        if not np.array_equal(self.symbols, item.symbols):
            return False
        if not np.array_equal(self.gaps, item.gaps):
            return False
        if not self.alphabet == item.alphabet:
            return False
        return True

    @staticmethod
    def from_alignment(alignment, alphabet=None):
        """
        Get an object of :class:`SequenceProfile` from an object of
        :class:`Alignment`.

        Based on the sequences of the alignment, the SequenceProfile
        parameters symbols and gaps are calculated.

        Parameters
        ----------
        alignment : Alignment
            An Alignment object to create the SequenceProfile object
            from.
        alphabet : bool
            This alphabet will be used when creating the SequenceProfile
            object. If no alphabet is selected, the alphabet for this
            :class:`SequenceProfile`.
            object will be calculated from the sequences of object
            Alignment.

        Returns
        -------
        profile: SequenceProfile
            The created :class:`SequenceProfile` object.
        """
        sequences = get_codes(alignment)
        if alphabet is None:
            alphabet = _determine_common_alphabet(
                [seq.alphabet for seq in alignment.sequences]
            )
        else:
            for alph in (seq.alphabet for seq in alignment.sequences):
                if not alphabet.extends(alph):
                    raise ValueError(
                        "The given alphabet is incompatible with a least one "
                        "alphabet of the given sequences"
                    )
        symbols = np.zeros((len(sequences[0]), len(alphabet)), dtype=int)
        gaps = np.zeros(len(sequences[0]), dtype=int)
        sequences = np.transpose(sequences)
        for i in range(len(sequences)):
            row = np.where(sequences[i,] == -1, len(alphabet), sequences[i,])
            count = np.bincount(row, minlength=len(alphabet) + 1)
            symbols[i,] = count[0 : len(alphabet)]
            gaps[i] = count[-1]
        return SequenceProfile(symbols, gaps, alphabet)

    def to_consensus(self, as_general=False):
        """
        Get the consensus sequence for this SequenceProfile object.

        Parameters
        ----------
        as_general : bool
            If true, returns consensus sequence as GeneralSequence
            object.
            Otherwise, the consensus sequence object type is chosen
            based on the alphabet of this SequenceProfile object.

        Returns
        -------
        consensus: Sequence
            The calculated consensus sequence.
        """
        # https://en.wikipedia.org/wiki/International_Union_of_Pure_and_Applied_Chemistry#Amino_acid_and_nucleotide_base_codes
        if as_general:
            return self._general_to_consensus()
        elif self.alphabet == _NUC_DNA_ALPH:
            return NucleotideSequence(self._dna_to_consensus())
        elif self.alphabet == _NUC_RNA_ALPH:
            return NucleotideSequence(self._rna_to_consensus())
        elif self.alphabet == _PROT_ALPH:
            return self._prot_to_consensus()
        return self._general_to_consensus()

    def _dna_to_consensus(self):
        codes = {
            (0,): "A",
            (1,): "C",
            (2,): "G",
            (3,): "T",
            (0, 2): "R",
            (1, 3): "Y",
            (1, 2): "S",
            (0, 3): "W",
            (2, 3): "K",
            (0, 1): "M",
            (1, 2, 3): "B",
            (0, 2, 3): "D",
            (0, 1, 3): "H",
            (0, 1, 2): "V",
            (0, 1, 2, 3): "N",
        }
        consensus = ""
        maxes = np.max(self.symbols, axis=1)
        for i in range(len(self.symbols)):
            consensus += _codes_to_iupac(self.symbols[i, :], codes, maxes[i], i)
        return consensus

    def _rna_to_consensus(self):
        codes = {
            (0,): "A",
            (1,): "C",
            (2,): "G",
            (3,): "U",
            (0, 2): "R",
            (1, 3): "Y",
            (1, 2): "S",
            (0, 3): "W",
            (2, 3): "K",
            (0, 1): "M",
            (1, 2, 3): "B",
            (0, 2, 3): "D",
            (0, 1, 3): "H",
            (0, 1, 2): "V",
            (0, 1, 2, 3): "N",
        }
        consensus = ""
        maxes = np.max(self.symbols, axis=1)
        for i in range(len(self.symbols)):
            consensus += _codes_to_iupac(self.symbols[i, :], codes, maxes[i], i)
        return consensus

    def _prot_to_consensus(self):
        """
        In case there is more than one symbol with the same maximal
        occurrences, the alphabetically sorted first symbol will be
        taken for the consensus sequence.
        """
        consensus = ProteinSequence()
        consensus.code = np.argmax(self.symbols, axis=1)
        consensus.code = np.where(
            np.sum(self.symbols, axis=1) == 0, 23, consensus.code
        )  # _PROT_ALPH[23] = 'X'
        return consensus

    def _general_to_consensus(self):
        """
        In case there is more than one symbol with the same maximal
        occurrences, the alphabetically sorted first symbol will be
        taken for the consensus sequence.
        In case the sum of occurrences of all symbols at a position is
        zero, the alphabetically sorted first symbol will be taken for
        the consensus sequence.
        """
        consensus = GeneralSequence(self.alphabet)
        consensus.code = np.argmax(self.symbols, axis=1)
        return consensus

    def probability_matrix(self, pseudocount=0):
        r"""
        Calculate the position probability matrix (PPM) based on
        'symbols' and the given pseudocount.
        This new matrix has the same shape as 'symbols'.

        .. math::

            P(S) = \frac {C_S + \frac{c_p}{k}} {\sum_{i} C_i + c_p}

        :math:`S`: The symbol.

        :math:`C_S`: The count of symbol :math:`S` at the sequence
        position.

        :math:`c_p`: The pseudocount.

        :math:`k`: Length of the alphabet.

        Parameters
        ----------
        pseudocount : int, optional
            Amount added to the number of observed cases in order to
            change the expected probability of the PPM.

        Returns
        -------
        ppm : ndarray, dtype=float, shape=(n,k)
            The calculated the position probability matrix.
        """
        if pseudocount < 0:
            raise ValueError("Pseudocount can not be smaller than zero.")
        return (self.symbols + pseudocount / self.symbols.shape[1]) / (
            np.sum(self.symbols, axis=1)[:, np.newaxis] + pseudocount
        )

    def log_odds_matrix(self, background_frequencies=None, pseudocount=0):
        r"""
        Calculate the position weight matrix (PWM) based on the
        position probability matrix (PPM) (with given pseudocount) and
        background_frequencies.
        This new matrix has the same shape as 'symbols'.

        .. math::

            W(S) = \log_2 \left( \frac{P(S)}{B_S} \right)

        :math:`S`: The symbol.

        :math:`P(S)`: The probability of symbol :math:`S` at the
        sequence position.

        :math:`c_p`: The background frequency of symbol :math:`S`.

        Parameters
        ----------
        background_frequencies : ndarray, shape=(k,), dtype=float, optional
            The background frequencies for each symbol in the alphabet.
            By default, a uniform distribution is assumed.
        pseudocount : int, optional
            Amount added to the number of observed cases in order to change
            the expected probability of the PPM.

        Returns
        -------
        pwm : ndarray, dtype=float, shape=(n,k)
            The calculated the position weight matrix.
        """
        if background_frequencies is None:
            background_frequencies = 1 / len(self.alphabet)
        ppm = self.probability_matrix(pseudocount=pseudocount)
        # Catch warning that appears, if a symbol is missing at any
        # position in the profile
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return np.log2(ppm / background_frequencies)

    def sequence_probability(self, sequence, pseudocount=0):
        r"""
        Calculate probability of a sequence based on the
        position probability matrix (PPM).

        The sequence probability is the product of the probability of
        the respective symbol over all sequence positions.

        Parameters
        ----------
        sequence : Sequence
           The input sequence.
        pseudocount : int, optional
            Amount added to the number of observed cases in order to change
            the expected probability of the PPM.

        Returns
        -------
        probability : float
           The calculated probability for the input sequence based on
           the PPM.
        """
        ppm = self.probability_matrix(pseudocount=pseudocount)
        if len(sequence) != len(ppm):
            raise ValueError(
                f"The given sequence has a different length ({len(sequence)}) than "
                f"the position probability matrix ({len(ppm)})."
            )
        if not ppm.shape == self.symbols.shape:
            raise ValueError(
                f"Position probability matrix {ppm.shape} must be of same shape "
                f"as 'symbols' {self.symbols.shape}"
            )
        return np.prod(ppm[np.arange(len(sequence)), sequence.code])

    def sequence_score(self, sequence, background_frequencies=None, pseudocount=0):
        """
        Calculate score of a sequence based on the
        position weight matrix (PWM).

        The score is the sum of weights (log-odds scores) of
        the respective symbol over all sequence positions.

        Parameters
        ----------
        sequence : Sequence
           The input sequence.
        background_frequencies : ndarray, shape=(k,), dtype=float, optional
            The background frequencies for each symbol in the alphabet.
            By default a uniform distribution is assumed.
        pseudocount : int, optional
            Amount added to the number of observed cases in order to change
            the expected probability of the PPM.

        Returns
        -------
        score : float
           The calculated score for the input sequence based on
           the PWM.
        """
        if background_frequencies is None:
            background_frequencies = 1 / len(self.alphabet)
        pwm = self.log_odds_matrix(
            background_frequencies=background_frequencies, pseudocount=pseudocount
        )
        if len(sequence) != len(pwm):
            raise ValueError(
                f"The given sequence has a different length ({len(sequence)}) than "
                f"the position weight matrix ({len(pwm)})."
            )
        if not pwm.shape == self.symbols.shape:
            raise ValueError(
                f"Position weight matrix {pwm.shape} must be of same shape "
                f"as 'symbols' {self.symbols.shape}"
            )
        return np.sum(pwm[np.arange(len(sequence)), sequence.code])

    def __getitem__(self, index):
        if isinstance(index, Integral):
            # Do not allow to collapse dimensions
            index = slice(index, index + 1)
        return SequenceProfile(self.symbols[index], self.gaps[index], self.alphabet)

    def __len__(self):
        return len(self.symbols)
