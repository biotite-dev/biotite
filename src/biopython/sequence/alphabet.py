# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import copy

__all__ = ["Alphabet", "AlphabetMapper", "AlphabetError"]


class Alphabet(object):
    """
    This class defines the allowed symbols for a `Sequence` and handles
    the encoding/decoding between symbols and symbol codes.
    
    An `Alphabet` is created with the list of symbols, that can be used
    in this context. In most cases a symbol will be simply a letter,
    hence a string of length 1. But in principal every hashable Python
    object can serve as symbol.
    
    The encoding of a symbol into a symbol code is
    done in the following way: Find the first index in the symbol list,
    where the list element equals the symbol. This index is the
    symbol code. If the symbol is not found in the list, an
    `AlphabetError` is raised.
    
    Internally, a dictionary is used for encoding, with symbols as keys
    and symbol codes as values. Therefore, every symbol must be
    hashable. For decoding the symbol list is indexed with the symbol
    code.
    
    If an alphabet *1* contains the same symbols and the same
    symbol-code-mappings like another alphabet *2*, but alphabet *1*
    introdues also new symbols, then alphabet *1* *extends* alphabet *2*.
    
    Objects of this class are immutable.
    
    Parameters
    ----------
    symbols : iterable object, optional
        The symbols, that are allowed in this alphabet. The
        corresponding code for a symbol, is the index of that symbol
        in this list.
    
    Examples
    --------
    """
    
    def __init__(self, symbols):
        self._symbols = copy.deepcopy(list(symbols))
        self._symbol_dict = {}
        for i, symbol in enumerate(symbols):
            self._symbol_dict[symbol] = i
    
    def get_symbols(self):
        return copy.deepcopy(self._symbols)
    
    def is_letter_alphabet(self):
        for symbol in self._symbols:
            if type(symbol) != string or len(symbol) != 1:
                return False
        return True
    
    def extends(self, alphabet):
        # Check for every symbol in the parent alphabet
        # if the symbol is also is the extending (this) alphabet
        # and has the same code (list index) for each symbol
        for i, symbol in enumerate(alphabet._symbols):
            if self._symbols[i] != symbol:
                return False
        return True
    
    def encode(self, symbol):
        try:
            return self._symbol_dict[symbol]
        except KeyError:
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
    

class AlphabetMapper(object):
    
    def __init__(source_alphabet, target_alphabet):
        self._mapper = [-1] * len(source_alphabet)
        for i in range(len(source_alphabet)):
            symbol = source_alphabet.decode(i)
            new_code = target_alphabet.encode(symbol)
            self._mapper[i] = new_code
        
    def __getitem__(code):
        return self._mapper[code]


class AlphabetError(Exception):
    pass