# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import copy

__all__ = ["Alphabet", "AlphabetError"]


class Alphabet(object):
    
    def __init__(self, symbols, parents=[]):
        self._symbols = copy.deepcopy(list(symbols))
        self._parents = copy.copy(list(parents))
    
    def get_symbols(self):
        return copy.deepcopy(self._symbols)
    
    def parent_alphabets(self):
        return copy.copy(self._parents)
    
    def extends(self, alphabet):
        return alphabet == self or alphabet in self._parents
    
    def encode(self, symbol):
        try:
            return self._symbols.index(symbol)
        except ValueError:
            raise AlphabetError(str(symbol) + " is not in the alphabet")
    
    def decode(self, code):
        try:
            return self._symbols[code]
        except IndexError:
            raise AlphabetError(str(code) + " is not a valid alphabet code")
    
    def __str__(self):
        return str(self._symbols)
    
    def __len__(self):
        return len(self._symbols)
    
    def __iter__(self):
        i = 0
        while i < len(self):
            yield self._symbols[i]
            i += 1
    

class AlphabetError(Exception):
    pass