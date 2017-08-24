# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
The module contains the `Sequence` superclass and `GeneralSequence`.
"""

import numpy as np
import abc
from .alphabet import Alphabet

__all__ = ["Sequence", "GeneralSequence"]


class Sequence(metaclass=abc.ABCMeta):
    """
    The abstract base class for all sequence types.
    
    A `Sequence` can be seen as a succession of symbols, that are
    elements in the allowed set of symbols, the `Alphabet`. Internally,
    a `Sequence` object uses a `numpy` `ndarray` of integers, where each
    integer represents a symbol. The `Alphabet` of a `Sequence` object
    is used to encode each symbol, that is used to create the
    `Sequence`, into an integer. These integer values are called
    *symbol code*, the encoding of an entire sequence of symbols is
    called *sequence code*.
    
    The size of the *symbol code* type in the array is determined by the 
    size of the `Alphabet`: If the `Alphabet` contains 256 symbols or
    less, one byte is used per array element; if the `Alphabet` contains
    between 257 and 65536 symbols, two bytes are used, and so on.
    
    Two `Sequence` objects are equal if they are instances of the same
    class, have the same `Alphabet` and have equal *sequence codes*.
    Comparison with a string or list of symbols evaluates always to
    false.
    
    A `Sequence` can be indexed by any 1-D index a `ndarray` accepts.
    If the index is a single integer, the decoded symbol at that
    position is returned, otherwise a subsequence is returned.
    Concatenation of two sequences is achieved with the '+' operator.
    
    Each subclass of `Sequence` needs to overwrite the abstract method
    `get_alphabet()`, which specifies the alphabet the `Sequence` uses.
    
    Parameters
    ----------
    sequence : iterable object, optional
        The symbol sequence, the `Sequence` is initialized with. For
        `Alphabets` containing single letter strings, this parameter
        may also be a `str` object. By default the sequence is empty.
    
    Examples
    --------
    Creating a DNA sequence from string and print the sequence and the
    sequence code:
    
        >>> dna_seq = DNASequence("ACGTA")
        >>> print(dna_seq)
        ACGTA
        >>> print(dna_seq.get_seq_code())
        [0 1 2 3 0]
    
    Sequence indexing:
        
        >>> print(dna_seq[1:3])
        CG
        >>> print(dna_seq[[0,2,4]])
        AGA
        
    Reverse the sequence:
        
        >>> dna_seq_rev = dna_seq.reverse()
        >>> print(dna_seq_rev)
        ATGCA
    
    Concatenate the two sequences:
        
        >>> dna_seq_concat = dna_seq + dna_seq_rev
        >>> print(dna_seq_concat)
        ACGTAATGCA
        
    """
    
    def __init__(self, sequence=[]):
        self.set_sequence(sequence)
    
    def copy(self, new_seq_code=None):
        seq_copy = type(self)()
        self._copy_code(seq_copy, new_seq_code)
        return seq_copy
    
    def _copy_code(self, new_object, new_seq_code):
        if new_seq_code is None:
            new_object.set_seq_code(self.get_seq_code())
        else:
            new_object.set_seq_code(new_seq_code)
    
    def set_sequence(self, sequence):
        self._seq_code = Sequence.encode(sequence, self.get_alphabet())
    
    def get_sequence(self):
        return Sequence.decode(code, self.get_alphabet())
    
    def set_seq_code(self, code):
        self._seq_code = code.astype(Sequence._dtype(len(self.get_alphabet())))
        
    def get_seq_code(self):
        return self._seq_code
    
    @abc.abstractmethod
    def get_alphabet(self):
        pass
    
    def reverse(self):
        reversed_code = np.flip(np.copy(self._seq_code), axis=0)
        reversed = self.copy(reversed_code)
        return reversed
    
    def __getitem__(self, index):
        alph = self.get_alphabet()
        sub_seq = self._seq_code.__getitem__(index)
        if isinstance(sub_seq, np.ndarray):
            return self.copy(sub_seq)
        else:
            return alph.decode(sub_seq)
    
    def __setitem__(self, index, symbol):
        alph = self.get_alphabet()
        symbol = alph.encode(symbol)
        self._seq_code.__setitem__(index, symbol)
    
    def __delitem__(self, index):
        self._seq_code = np.delete(self._seq_code, index) 
    
    def __len__(self):
        return len(self._seq_code)
    
    def __iter__(self):
        alph = self.get_alphabet()
        i = 0
        while i < len(self):
            yield alph.decode(self._seq_code[i])
            i += 1
    
    def __eq__(self, item):
        if not isinstance(item, type(self)):
            return False
        if self.get_alphabet() != item.get_alphabet():
            return False
        return np.array_equal(self._seq_code, item._seq_code)
    
    def __ne__(self, item):
        return not self == item
    
    def __str__(self):
        alph = self.get_alphabet()
        string = ""
        for e in self._seq_code:
            string += alph.decode(e)
        return string
    
    def __add__(self, sequence):
        if self.get_alphabet().extends(sequence.get_alphabet()):
            new_code = np.concatenate((self._seq_code, sequence._seq_code))
            new_seq = self.copy(new_code)
            return new_seq
        elif sequence.get_alphabet().extends(self.get_alphabet()):
            pass
            new_code = np.concatenate((self._seq_code, sequence._seq_code))
            new_seq = sequence.copy(new_code)
            return new_seq
        else:
            raise ValueError("The sequences alphabets are not compatible")
    
    @staticmethod
    def encode(sequence, alphabet):
        return np.array([alphabet.encode(e) for e in sequence],
                        dtype=Sequence._dtype(len(alphabet)))

    @staticmethod
    def decode(seq_code, alphabet):
        return [alphabet.decode(code) for code in seq_code]

    @staticmethod
    def _dtype(alphabet_size):
        byte_count = 1
        while 256**byte_count < alphabet_size:
            i += 1
        return "u{:d}".format(byte_count)


class GeneralSequence(Sequence):
    
    def __init__(self, alphabet, sequence=[]):
        self._alphabet = alphabet
        super().__init__(sequence)
    
    def copy(self, new_seq_code=None):
        seq_copy = GeneralSequence(self._alphabet)
        self._copy_code(seq_copy, new_seq_code)
        return seq_copy
    
    def get_alphabet(self):
        return self._alphabet
