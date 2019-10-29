# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["WordAlphabet", "LetterWordAlphabet", "WordSequence"]

import copy
import string
import numpy as np
from ..sequence import Sequence
from ..alphabet import Alphabet, LetterAlphabet, AlphabetError
from .wordconvert import combine_into_words



class WordAlphabet(Alphabet):
    
    def __init__(self, base_alphabet, word_length):
        if not isinstance(base_alphabet, Alphabet):
            raise TypeError(
                f"Got {type(base_alphabet).__name__}, "
                f"but Alphabet was expected"
            )
        if word_length < 2:
            raise ValueError("Word length must be at least 2")
        self._base_alph = base_alphabet
        self._word_length = word_length
        self._base_alph_symbols = self._base_alph.get_symbols()
        self._radix_multiplier = np.array(
            [len(self._base_alph)**n for n in range(self._word_length)],
            dtype=int
        )
    
    def get_base_alphabet(self):
        return self._base_alph
    
    def get_word_length(self):
        return self._word_length
    
    def get_symbols(self):
        return [self.decode(code) for code in range(len(self))]
    
    def extends(self, alphabet):
        # A WordAlphabet cannot really extend another WordAlphabet:
        # If the word length is not equal all symbols are not equal
        # If the base alphabet has additional symbols, the correct
        # order is not preserved
        # The only way that WordAlphabet 'extends' another WordAlphabet,
        # if the two alphabets are equal
        if alphabet is self:
            return True
        if alphabet == self:
            return True
        return False
    
    def encode(self, symbol):
        if len(symbol) != self._word_length:
            raise AlphabetError(
                f"Symbol has word length {len(symbol)}, "
                f"but alphabet expects {self._word_length}"
            )
        
        code = 0
        for pos in range(len(symbol)):
            try:
                code += self._radix_multiplier[pos] \
                        * self._base_alph_symbols.index(symbol[pos])
            except ValueError:
                raise AlphabetError(f"'{symbol}' is not in the alphabet")
        return code
    
    def decode(self, code):
        if code >= len(self) or code < 0:
            raise AlphabetError(
                f"Symbol code {code} is invalid for this alphabet"
            )
        word = [None] * self._word_length
        for n in reversed(range(self._word_length)):
            val = self._radix_multiplier[n]
            alph_index = code // val
            word[-(n+1)] = self._base_alph_symbols[alph_index]
            code = code - alph_index * val
        return tuple(word)
    
    def combine_into_words(self, base_code):
        if len(base_code) < self._word_length:
            raise ValueError("Sequence code is shorter than word length")
        word_code = np.empty(
            len(base_code) - self._word_length + 1,
            dtype=Sequence.dtype(len(self))
        )
        combine_into_words(
            base_code, word_code, self._word_length, len(self._base_alph)
        )
        return word_code
    
    def __str__(self):
        return str(self.get_symbols())
    
    def __eq__(self, item):
        if not isinstance(item, WordAlphabet):
            return False
        if self._base_alph != item._base_alph:
            return False
        if self._word_length != item._word_length:
            return False
        return True
    
    def __len__(self):
        return int(len(self._base_alph) ** self._word_length)



class LetterWordAlphabet(WordAlphabet):
    
    def __init__(self, base_alphabet, word_length):
        if not isinstance(base_alphabet, LetterAlphabet):
            raise TypeError(
                f"Got {type(base_alphabet).__name__}, "
                f"but LetterAlphabet was expected"
            )
        super().__init__(base_alphabet, word_length)
    
    def decode(self, code):
        return "".join(super().decode(code))
    
    def __eq__(self, item):
        if not isinstance(item, LetterWordAlphabet):
            return False
        return super().__eq__(item)



class WordSequence(Sequence):

    def __init__(self, base_sequence, word_length):
        base_alphabet = base_sequence.get_alphabet()
        if isinstance(base_alphabet, LetterAlphabet):
            self._alphabet = LetterWordAlphabet(base_alphabet, word_length)
        else:
            self._alphabet = WordAlphabet(base_alphabet, word_length)
        self.code = self._alphabet.combine_into_words(base_sequence.code)
    
    def get_alphabet(self):
        return self._alphabet