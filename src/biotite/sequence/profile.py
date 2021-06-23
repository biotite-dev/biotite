# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
from .seqtypes import NucleotideSequence, ProteinSequence, GeneralSequence
from .alphabet import LetterAlphabet
from .align.alignment import get_codes

__name__ = "biotite.sequence"
__author__ = "Maximilian Greil"
_all__ = ["SequenceProfile"]

# Abbreviations
_NUC_DNA_ALPH = NucleotideSequence.alphabet_unamb
_NUC_RNA_ALPH = LetterAlphabet(["A", "C", "G", "U"])
_PROT_ALPH = ProteinSequence.alphabet


class SequenceProfile(object):
    """
    A :class:`SequenceProfile` object stores information about a sequence
    profile of aligned sequences. It is possible to calculate and return
    its consensus sequence.

    This class saves the position frequency matrix (symbols) of the occurrences of
    each alphabet symbol at each position. It also saves the number of gaps
    at each position (gaps).

    With method from_alignment() a :class:`SequenceProfile` object can be
    created from an indefinite number of aligned sequences.

    All attributes of this class are publicly accessible.

    Parameters
    ----------
    symbols : ndarray, dtype=int, shape=(n,k)
        This matrix simply saves for each position how often absolutely each symbol is present.
    gaps : ndarray, dtype=int, shape=n
        Array which indicates the number of gaps at each position.
    alphabet : Alphabet, length=k
        Alphabet of sequences of sequence profile

    Attributes
    ----------
    symbols : ndarray, dtype=int, shape=(n,k)
        This matrix simply saves for each position how often absolutely each symbol is present.
    gaps : ndarray, dtype=int, shape=n
        Array which indicates the number of gaps at each position.
    alphabet : Alphabet, length=k
        Alphabet of sequences of sequence profile
    """

    def __init__(self, symbols, gaps, alphabet):
        self._symbols = symbols
        self._gaps = gaps
        self._alphabet = alphabet

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
            raise IndexError(f"New symbols ndarray must be of same shape {self.symbols.shape} as old one")
        self._symbols = new_symbols

    @gaps.setter
    def gaps(self, new_gaps):
        if not new_gaps.shape == self.gaps.shape:
            raise IndexError(f"New gaps ndarray must be of same shape {self.gaps.shape} as old one")
        self._gaps = new_gaps

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
    def from_alignment(alignment):
        """
        Get an object of :class:`SequenceProfile` from an object of :class:`Alignment`.

        Based on the sequences of the alignment, the SequenceProfile parameters symbols
        and gaps are calculated.
        """
        sequences = get_codes(alignment)
        for i in range(1, len(sequences)):
            if not set(alignment.sequences[i-1].get_alphabet().get_symbols()) == \
                   set(alignment.sequences[i].get_alphabet().get_symbols()):
                raise TypeError("Alignment contains sequences with different alphabets")
        alphabet = alignment.sequences[0].get_alphabet().get_symbols()
        symbols = np.zeros((len(sequences[0]), len(alphabet)), dtype=int)
        gaps = np.zeros(len(sequences[0]), dtype=int)
        sequences = np.transpose(sequences)
        for i in range(len(sequences)):
            row = np.where(sequences[i, ] == -1, len(alphabet), sequences[i, ])
            if len(np.bincount(row)) < len(alphabet):
                symbols[i, 0:len(np.bincount(row))] = np.bincount(row)
            elif len(np.bincount(row)) > len(alphabet):
                symbols[i, ] = np.bincount(row)[0:len(alphabet)]
                gaps[i] = np.bincount(row)[-1]
            else:
                symbols[i, ] = np.bincount(row)
        return SequenceProfile(symbols, gaps, alignment.sequences[0].get_alphabet())

    def to_consensus(self, as_general=False):
        """
        Get the consensus sequence for this SequenceProfile object.

        Parameters
        ----------
        as_general : bool
            If true, returns consensus sequence as GeneralSequence object.
            Otherwise, the consensus sequence object type is chosen based
            on the alphabet of this SequenceProfile object.(Default: False).

        Returns
        -------
        consensus: either NucleotideSequence, ProteinSequence or GeneralSequence
            The calculated consensus sequence
        """
        # https://en.wikipedia.org/wiki/International_Union_of_Pure_and_Applied_Chemistry#Amino_acid_and_nucleotide_base_codes
        if as_general:
            return GeneralSequence(self.alphabet, self._custom_to_consensus())
        elif set(self.alphabet.get_symbols()) == set(_NUC_DNA_ALPH.get_symbols()):
            return NucleotideSequence(self._dna_to_consensus())
        elif set(self.alphabet.get_symbols()) == set(_NUC_RNA_ALPH.get_symbols()):
            return NucleotideSequence(self._rna_to_consensus())
        elif set(self.alphabet.get_symbols()) == set(_PROT_ALPH.get_symbols()):
            return ProteinSequence(self._prot_to_consensus())
        return GeneralSequence(self.alphabet, self._custom_to_consensus())

    def _codes_to_iupac(self, frequency, codes):
        if np.sum(frequency) == 0:
            return "-"
        position_max = np.where(frequency == np.amax(frequency))[0].tolist()
        key = "".join(["".join(self.alphabet.get_symbols()[i]) for i in position_max])
        return codes[key]

    def _dna_to_consensus(self):
        codes = {
            'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T',
            'AG': 'R', 'CT': 'Y', 'CG': 'S', 'AT': 'W', 'GT': 'K', 'AC': 'M',
            'CGT': 'B', 'AGT': 'D', 'ACT': 'H', 'ACG': 'V',
            'ACGT': 'N'
        }
        consensus = ""
        for i in range(len(self.symbols)):
            consensus += self._codes_to_iupac(self.symbols[i, :], codes)
        return consensus

    def _rna_to_consensus(self):
        codes = {
            'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U',
            'AG': 'R', 'CU': 'Y', 'CG': 'S', 'AU': 'W', 'GU': 'K', 'AC': 'M',
            'CGU': 'B', 'AGU': 'D', 'ACU': 'H', 'ACG': 'V',
            'ACGU': 'N'
        }
        consensus = ""
        for i in range(len(self.symbols)):
            consensus += self._codes_to_iupac(self.symbols[i, :], codes)
        return consensus

    def _prot_to_consensus(self):
        # In case there is more than one symbol with the same maximal occurrences, the alphabetically sorted first
        # symbol will be taken for the consensus sequence
        consensus = ""
        for i in range(len(self.symbols)):
            if np.sum(self.symbols[i, :]) == 0:
                consensus += "-"
            elif np.sum(self.symbols[i, :]) / len(self.symbols[i, :]) == 1:
                consensus += "X"
            else:
                position_max = np.where(self.symbols[i, :] ==
                                        np.amax(self.symbols[i, :]))[0].tolist()
                consensus += self.alphabet.get_symbols()[position_max[0]]
        return consensus

    def _custom_to_consensus(self):
        # In case there is more than one symbol with the same maximal occurrences, the alphabetically sorted first
        # symbol will be taken for the consensus sequence
        consensus = ""
        for i in range(len(self.symbols)):
            if np.sum(self.symbols[i, :]) == 0:
                consensus += "-"
            else:
                position_max = np.where(self.symbols[i, :] ==
                                        np.amax(self.symbols[i, :]))[0].tolist()
                consensus += self.alphabet.get_symbols()[position_max[0]]
        return consensus
