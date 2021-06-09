# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
from .seqtypes import NucleotideSequence, ProteinSequence
from .alphabet import LetterAlphabet

__name__ = "biotite.sequence"
__author__ = "Maximilian Greil"
_all__ = ["SequenceProfile", "to_consensus"]

# Abbreviations
_NUC_DNA_ALPH = NucleotideSequence.alphabet_unamb
_NUC_RNA_ALPH = LetterAlphabet(["A", "C", "G", "U"])
_PROT_ALPH = ProteinSequence.alphabet


class SequenceProfile(object):
    """
    A :class:`SequenceProfile` object stores information about a sequence
    profile of aligned sequences, i.e. the consensus sequence.

    This class saves the position frequency matrix of the occurrences of
    each alphabet symbol at each position. It also saves the number of gaps
    at each position.

    A :class:`SequenceProfile` object can be created from an indefinite number
    of aligned sequences.

    All attributes of this class are publicly accessible.

    Parameters
    ----------
    position_frequency_matrix : ndarray, dtype=int, shape=(n,k)
        This matrix simply saves for each position how often absolutely each symbol is present.
    position_gaps : ndarray, dtype=int, shape=n
        Array which indicates the number of gaps at each position.
    alphabet : Alphabet, length=k
        Alphabet of sequences of sequence profile

    Attributes
    ----------
    position_frequency_matrix : ndarray, dtype=int, shape=(n,k)
        This matrix simply saves for each position how often absolutely each symbol is present.
    position_gaps : ndarray, dtype=int, shape=n
        Array which indicates the number of gaps at each position.
    alphabet : Alphabet, length=k
        Alphabet of sequences of sequence profile
    """

    def __init__(self, position_frequency_matrix, position_gaps, alphabet):
        self.position_frequency_matrix = position_frequency_matrix
        self.position_gaps = position_gaps
        self.alphabet = alphabet

    def __repr__(self):
        """Represent SequenceProfile as a string for debugging."""
        return f"SequenceProfile(np.{np.array_repr(self.position_frequency_matrix)}, " \
               f"np.{np.array_repr(self.position_gaps)}, Alphabet({self.alphabet}))"

    def __eq__(self, item):
        if not isinstance(item, SequenceProfile):
            return False
        if not np.array_equal(self.position_frequency_matrix, item.position_frequency_matrix):
            return False
        if not np.array_equal(self.position_gaps, item.position_gaps):
            return False
        if not self.alphabet == item.alphabet:
            return False
        return True

    @staticmethod
    def from_alignment(alignment):
        """
        Get an object of :class:`SequenceProfile` from an object of :class:`Alignment`
        """
        sequences = alignment.get_gapped_sequences()
        alphabet = alignment.sequences[0].get_alphabet().get_symbols()
        position_frequency_matrix = np.zeros((len(sequences[0]), len(alphabet)), dtype=int)
        position_gaps = np.zeros(len(sequences[0]), dtype=int)
        for i in range(len(sequences[0])):
            for j in range(len(sequences)):
                if sequences[j][i] != "-":
                    position_frequency_matrix[i, alphabet.index(sequences[j][i])] += 1
                else:
                    position_gaps[i] += 1
        return SequenceProfile(position_frequency_matrix, position_gaps, alignment.sequences[0].get_alphabet())

    def to_consensus(self):
        """
        Get the consensus sequence for :class:`SequenceProfile`

        Returns
        -------
        consensus: str
            The calculated consensus sequence
        """
        # https://en.wikipedia.org/wiki/International_Union_of_Pure_and_Applied_Chemistry#Amino_acid_and_nucleotide_base_codes
        if set(self.alphabet.get_symbols()) == set(_NUC_DNA_ALPH.get_symbols()):
            return self._dna_to_consensus()
        elif set(self.alphabet.get_symbols()) == set(_NUC_RNA_ALPH.get_symbols()):
            return self._rna_to_consensus()
        elif set(self.alphabet.get_symbols()) == set(_PROT_ALPH.get_symbols()):
            return self._prot_to_consensus()
        return self._custom_to_consensus()

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
        for i in range(len(self.position_frequency_matrix)):
            consensus += self._codes_to_iupac(self.position_frequency_matrix[i, :], codes)
        return consensus

    def _rna_to_consensus(self):
        codes = {
            'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U',
            'AG': 'R', 'CU': 'Y', 'CG': 'S', 'AU': 'W', 'GU': 'K', 'AC': 'M',
            'CGU': 'B', 'AGU': 'D', 'ACU': 'H', 'ACG': 'V',
            'ACGU': 'N'
        }
        consensus = ""
        for i in range(len(self.position_frequency_matrix)):
            consensus += self._codes_to_iupac(self.position_frequency_matrix[i, :], codes)
        return consensus

    def _prot_to_consensus(self):
        # In case there is more than one symbol with the same maximal occurrences, the alphabetically sorted first
        # symbol will be taken for the consensus sequence
        consensus = ""
        for i in range(len(self.position_frequency_matrix)):
            if np.sum(self.position_frequency_matrix[i, :]) == 0:
                consensus += "-"
            elif np.sum(self.position_frequency_matrix[i, :]) / len(self.position_frequency_matrix[i, :]) == 1:
                consensus += "X"
            else:
                position_max = np.where(self.position_frequency_matrix[i, :] ==
                                        np.amax(self.position_frequency_matrix[i, :]))[0].tolist()
                consensus += self.alphabet.get_symbols()[position_max[0]]
        return consensus

    def _custom_to_consensus(self):
        # In case there is more than one symbol with the same maximal occurrences, the alphabetically sorted first
        # symbol will be taken for the consensus sequence
        consensus = ""
        for i in range(len(self.position_frequency_matrix)):
            if np.sum(self.position_frequency_matrix[i, :]) == 0:
                consensus += "-"
            else:
                position_max = np.where(self.position_frequency_matrix[i, :] ==
                                        np.amax(self.position_frequency_matrix[i, :]))[0].tolist()
                consensus += self.alphabet.get_symbols()[position_max[0]]
        return consensus
