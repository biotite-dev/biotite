# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
from .seqtypes import NucleotideSequence, ProteinSequence, GeneralSequence
from .alphabet import LetterAlphabet
from .align.alignment import get_codes

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

    This class saves the position frequency matrix (position count matrix)
    'symbols' of the occurrences of each alphabet symbol at each position.
    It also saves the number of gaps at each position in the array
    'gaps'.

    Additionally, you can get the position probability matrix ('ppm') and the
    position weight matrix ('pmw') of an instance of object SequenceProfile.

    Matrix 'ppm' is calculated based on 'symbols' and parameter 'pseudocount'.
    Matrix 'pwm' is calculated based on 'ppm' and parameter
    'background_frequencies'.

    With :meth:`from_alignment()` a :class:`SequenceProfile` object can
    be created from an indefinite number of aligned sequences.

    With :meth:`sequence_probability_from_matrix()` the probability of a sequence
    can be calculated based on the 'ppm' or 'pmw' of this instance of
    object SequenceProfile.

    All attributes of this class are publicly accessible.

    Parameters
    ----------
    symbols : ndarray, dtype=int, shape=(n,k)
        This matrix simply saves for each position how often absolutely
        each symbol is present.
    gaps : ndarray, dtype=int, shape=n
        Array which indicates the number of gaps at each position.
    alphabet : Alphabet, length=k
        Alphabet of sequences of sequence profile

    Attributes
    ----------
    symbols : ndarray, dtype=int, shape=(n,k)
        This matrix simply saves for each position how often absolutely
        each symbol is present.
    gaps : ndarray, dtype=int, shape=n
        Array which indicates the number of gaps at each position.
    alphabet : Alphabet, length=k
        Alphabet of sequences of sequence profile
    ppm : ndarray, dtype=float, shape=(n,k)
        This matrix is the position probability matrix calculated based
        on symbols and pseudocount.
    pseudocount : int
        Amount added to the number of observed cases in order to change
        the expected probability.
    pwm : ndarray, dtype=float, shape=(n,k)
        This matrix is the position weight matrix calculated based
        on ppm and background_frequencies.
    background_frequencies : None
        To be implemented.
    """

    def __init__(self, symbols, gaps, alphabet):
        self._symbols = symbols
        self._gaps = gaps
        self._alphabet = alphabet
        self._ppm = None
        self._pseudocount = 0
        self._pwm = None
        self._background_frequencies = 1 / len(self.alphabet)  # uniform distribution

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

    @property
    def ppm(self):
        if self._ppm is None:
            self._ppm = self._probability_matrix()
        return self._ppm

    @property
    def pseudocount(self):
        return self._pseudocount

    @pseudocount.setter
    def pseudocount(self, new_pseudocount):
        if new_pseudocount < 0:
            raise ValueError(
                f"Pseudocount can not be smaller than zero."
            )
        self._pseudocount = new_pseudocount
        self._ppm = self._probability_matrix()
        self._pwm = self._log_odds_matrix()  # pwm dependent on ppm

    @property
    def pwm(self):
        if self._pwm is None:
            self._pwm = self._log_odds_matrix()
        return self._pwm

    @property
    def background_frequencies(self):
        return self._background_frequencies

    @background_frequencies.setter
    def background_frequencies(self, new_background_frequencies):
        self._background_frequencies = new_background_frequencies
        self._pwm = self._log_odds_matrix()

    def __repr__(self):
        """Represent SequenceProfile as a string for debugging."""
        return f"SequenceProfile(np.{np.array_repr(self.symbols)}, " \
               f"np.{np.array_repr(self.gaps)}, Alphabet({self.alphabet}))"

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
            SequenceProfile
            object will be calculated from the sequences of object
            Alignment.
            (Default: None).

        Returns
        -------
        profile: SequenceProfile
            The created SequenceProfile object
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
                        f"The given alphabet is incompatible with a least one "
                        "alphabet of the given sequences"
                    )
        symbols = np.zeros((len(sequences[0]), len(alphabet)), dtype=int)
        gaps = np.zeros(len(sequences[0]), dtype=int)
        sequences = np.transpose(sequences)
        for i in range(len(sequences)):
            row = np.where(sequences[i, ] == -1, len(alphabet), sequences[i, ])
            count = np.bincount(row, minlength=len(alphabet) + 1)
            symbols[i, ] = count[0:len(alphabet)]
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
            based on the alphabet of this SequenceProfile object
            (Default: False).

        Returns
        -------
        consensus: Sequence
            The calculated consensus sequence
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
            (0,): 'A', (1,): 'C', (2,): 'G', (3,): 'T',
            (0, 2): 'R', (1, 3): 'Y', (1, 2): 'S', (0, 3): 'W', (2, 3): 'K', (0, 1): 'M',
            (1, 2, 3): 'B', (0, 2, 3): 'D', (0, 1, 3): 'H', (0, 1, 2): 'V',
            (0, 1, 2, 3): 'N'
        }
        consensus = ""
        maxes = np.max(self.symbols, axis=1)
        for i in range(len(self.symbols)):
            consensus += _codes_to_iupac(self.symbols[i, :], codes, maxes[i], i)
        return consensus

    def _rna_to_consensus(self):
        codes = {
            (0,): 'A', (1,): 'C', (2,): 'G', (3,): 'U',
            (0, 2): 'R', (1, 3): 'Y', (1, 2): 'S', (0, 3): 'W', (2, 3): 'K', (0, 1): 'M',
            (1, 2, 3): 'B', (0, 2, 3): 'D', (0, 1, 3): 'H', (0, 1, 2): 'V',
            (0, 1, 2, 3): 'N'
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

    def _probability_matrix(self):
        return (self.symbols + self.pseudocount / len(self.symbols)) / \
               (np.sum(self.symbols, axis=1)[:, np.newaxis] + self.pseudocount)

    def _log_odds_matrix(self):
        return np.log2(self.ppm / self.background_frequencies)

    def sequence_probability_from_matrix(self, sequence, matrix="ppm"):
        """
        Calculate probability of a sequence based on either the
        position probability matrix (ppm) or the position weight matrix (pwm).

        Parameters
        ----------
        sequence : Sequence
           The input sequence.
        matrix : string
            The chosen position matrix to use to calculate the probability
            of the input sequence (either ppm or pwm).
            (Default: ppm).

        Returns
        -------
        probability: float
           The calculated probability for the input sequence based on
           the chosen position matrix.
        """
        if matrix != "ppm" and matrix != "pwm":
            raise ValueError(
                f'Input parameter matrix must be either "ppm" (position probability matrix) '
                f'or "pwm" (position weight matrix).'
            )

        probability = 1
        codes = sequence.code

        if matrix == "ppm":
            if len(sequence) != len(self.ppm):
                raise ValueError(
                    f"The given sequence has a different length ({len(sequence)}) than "
                    f"position probability matrix {len(self.ppm)}."
                )
            else:
                for i in range(len(codes)):
                    probability = probability * self.ppm[i, codes[i]]
        else:
            if len(sequence) != len(self.pwm):
                raise ValueError(
                    f"The given sequence has a different length ({len(sequence)}) than "
                    f"position weight matrix ({len(self.pwm)})."
                )
            else:
                for i in range(len(codes)):
                    probability = probability * self.pwm[i, codes[i]]
        return probability
