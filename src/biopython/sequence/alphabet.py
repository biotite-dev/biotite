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
    Per definition, every alphabet also extends itself.
    
    Objects of this class are immutable.
    
    Parameters
    ----------
    symbols : iterable object, optional
        The symbols, that are allowed in this alphabet. The
        corresponding code for a symbol, is the index of that symbol
        in this list.
    
    Examples
    --------
    Create an Alphabet containing DNA letters and encode/decode a
    letter/code:
    
        >>> alph = Alphabet(["A","C","G","T"])
        >>> print(alph.encode("G"))
        2
        >>> print(alph.decode(2))
        G
        >>> try:
        ...    alph.encode("foo")
        >>> except Exception as e:
        ...    print(e)
        'foo' is not in the alphabet
    
    Create an Alphabet of arbitrary objects:
    
        >>> alph = Alphabet(["foo", 42, (1,2,3), 5, 3.141])
        >>> print(alph.encode((1,2,3)))
        2
        >>> print(alph.decode(4))
        3.141
        
    """
    
    def __init__(self, symbols):
        if len(symbols) == 0:
            raise ValueError("Symbol list is empty")
        self._symbols = copy.deepcopy(list(symbols))
        self._symbol_dict = {}
        for i, symbol in enumerate(symbols):
            self._symbol_dict[symbol] = i
    
    def get_symbols(self):
        """
        Get the symbols in the alphabet.
        
        Returns
        -------
        symbols : list
            Copy of the internal list of symbols.
        """
        return copy.deepcopy(self._symbols)
    
    def is_letter_alphabet(self):
        """
        Check, if this alphabet only contains single letter symbols.
        
        Returns
        -------
        result : bool
            True, if this alphabet uses exclusively single letters
            symbols, false otherwise.
        """
        for symbol in self._symbols:
            if type(symbol) != string or len(symbol) != 1:
                return False
        return True
    
    def extends(self, alphabet):
        """
        Check, if this alphabet extends another alphabet.
        
        Parameters
        ----------
        alphabet : Alphabet
            The potential parent alphabet.
        
        Returns
        -------
        result : bool
            True, if this object extends `alphabet`, false otherwise.
        """
        if alphabet is self:
            return True
        # Check for every symbol in the parent alphabet
        # if the symbol is also is the extending (this) alphabet
        # and has the same code (list index) for each symbol
        for i, symbol in enumerate(alphabet._symbols):
            if self._symbols[i] != symbol:
                return False
        return True
    
    def encode(self, symbol):
        """
        Use the alphabet to encode a symbol.
        
        Parameters
        ----------
        symbol : object
            The object to encode into a symbol code.
        
        Returns
        -------
        code : int
            The symbol code of `symbol`.
        
        Raises
        ------
        AlphabetError
            If `symbol` is not in the alphabet.
        """
        try:
            return self._symbol_dict[symbol]
        except KeyError:
            raise AlphabetError("'" + str(symbol) + "' is not in the alphabet")
    
    def decode(self, code):
        """
        Use the alphabet to decode a symbol code.
        
        Parameters
        ----------
        code : int
            The symbol code to be decoded.
        
        Returns
        -------
        symbol : object
            The symbol corresponding to `code`.
        
        Raises
        ------
        AlphabetError
            If `code` is not a valid code in the alphabet.
        """
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
    """
    This class is used for symbol code conversion from a source
    alphabet into a target alphabet.
    
    Parameters
    ----------
    source_alphabet, target_alphabet : Alphabet
        The codes are converted from the source alphabet into the
        target alphabet.
    
    Examples
    --------
        
        >>> source_alph = Alphabet(["A","C","G","T"])
        >>> target_alph = Alphabet(["T","U","A","G","C"])
        >>> mapper = AlphabetMapper(source_alph, target_alph)
        >>> print(mapper[0])
        2
        >>> print(mapper[1])
        4
        >>> print(mapper[2])
        3
        >>> print(mapper[3])
        0
        
    """
    
    def __init__(self, source_alphabet, target_alphabet):
        self._mapper = [-1] * len(source_alphabet)
        for i in range(len(source_alphabet)):
            symbol = source_alphabet.decode(i)
            new_code = target_alphabet.encode(symbol)
            self._mapper[i] = new_code
        
    def __getitem__(self, code):
        return self._mapper[code]


class AlphabetError(Exception):
    """
    This exception is raised, when a code or a symbol is not in an
    `Alphabet`.
    """
    pass