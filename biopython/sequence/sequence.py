# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
import abc
from .alphabet import Alphabet

class Sequence(metaclass=abc.ABCMeta):
    
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
    
    def find_subsequence(sequence):
        if not self.get_alphabet.extends(sequence.get_alphabet()):
            raise ValueError("The sequences alphabets are not equal")
        match_indices = []
        frame_size = len(sequence)
        for i in range(len(self) - frame_size + 1):
            sub_seq = self._seq_code[i : i + frame_size]
            if np.array_equal(sequence, sub_seq):
                match_indices.append(i)
        return match_indices
    
    def find_symbol(symbol):
        code = self.get_alphabet().encode(symbol)
        return numpy.where(self._seq_code == code)
    
    def reverse(self):
        reversed_code = np.flip(np.copy(self._seq_code), axis=0)
        reversed = self.copy(reversed_code)
        return reversed
    
    def __getitem__(self, index):
        alph = get_alphabet(self)
        sub_seq = self._seq_code.__getitem__(index)
        if isinstance(item, np.ndarray):
            new_seq = sequence.copy(sub_seq)
        else:
            return alph.decode(sub_seq)
    
    def __setitem__(self, index, symbol):
        alph = get_alphabet(self)
        symbol = alph.encode(symbol)
        self._seq_code.__setitem__(index, symbol)
    
    def __delitem__(self, index):
        self._seq_code.__delitem__(index) 
    
    def __len__(self):
        return len(self._seq_code)
    
    def __iter__(self):
        alph = self.get_alphabet()
        i = 0
        while i < len(self):
            yield alph.decode(self._seq_code[i])
            i += 1
    
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
