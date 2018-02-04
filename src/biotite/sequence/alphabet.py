# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

import copy
import numpy as np

__all__ = ["Alphabet", "LetterAlphabet", "AlphabetMapper", "AlphabetError"]


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
        elif len(alphabet) > len(self):
            return False
        else:
            return list(alphabet._symbols) \
                == list(self._symbols[:len(alphabet)])
    
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
            raise AlphabetError("'{:}' is not in the alphabet"
                                .format(str(symbol)))
    
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
            raise AlphabetError("'{:}' is not a valid code"
                                .format(str(code)))
    
    def encode_multiple(self, symbols, dtype=np.int64):
        """
        Encode a list of symbols.
        
        Parameters
        ----------
        symbols : array-like
            The symbols to encode.
        dtype : dtype, optional
            The dtype of the output ndarray. (Default: `int64`)
            
        Returns
        -------
        code : ndarray
            The sequence code.
        """
        return np.array([self.encode(e) for e in symbols], dtype=dtype)
    
    def decode_multiple(self, code):
        """
        Decode a sequence code into a list of symbols.
        
        Parameters
        ----------
        code : ndarray
            The sequence code to decode.
        
        Returns
        -------
        symbols : list
            The decoded list of symbols.
        """
        return [self.decode(c) for c in code]
    
    def __str__(self):
        return str(self._symbols)
    
    def __len__(self):
        return len(self._symbols)
    
    def __iter__(self):
        i = 0
        while i < len(self):
            yield self._symbols[i]
            i += 1
    
    def __contains__(self, symbol):
        return symbol in self._symbols


class LetterAlphabet(Alphabet):
    
    def __init__(self, symbols):
        if len(symbols) == 0:
            raise ValueError("Symbol list is empty")
        if len(symbols) > 128:
            raise ValueError("Symbol list is too large")
        for symbol in symbols:
            if not isinstance(symbol, str) or len(symbol) > 1:
                raise ValueError("Symbol '{:}' is not a single letter"
                                 .format(str(symbol)))
        self._symbols = np.array(list(symbols), dtype="U1")
    
    def get_symbols(self):
        """
        Get the symbols in the alphabet.
        
        Returns
        -------
        symbols : ndarray
            Copy of the internal list of symbols.
        """
        return np.copy(self._symbols)
    
    def encode(self, symbol):
        indices = np.where(self._symbols == symbol)[0]
        if len(indices) == 0:
            raise AlphabetError("'{:}' is not in the alphabet"
                                .format(str(symbol)))
        return indices[0]
    
    def encode_multiple(self, symbols, dtype=None):
        """
        Encode a list of symbols.
        
        Parameters
        ----------
        symbols : array-like
            The symbols to encode. The method is faster when a
            `ndarray` is provided.
        dtype : dtype, optional
            For compatibility with superclass. The value is ignored
            
        Returns
        -------
        code : ndarray
            The sequence code.
        """
        if not isinstance(symbols, np.ndarray):
            symbols = np.array(list(symbols), dtype="U1")
        # Initially fill the sequence code
        # with the last allowed symbol code + 1
        # Since this code cannot occur from symbol encoding
        # it can be later used to check for illegal symbols
        illegal_code = len(self)
        code = np.full(len(symbols), illegal_code, dtype=np.uint8)
        # This is only efficient for short alphabets
        # Therefore it is only used in the LetterAlphabet class
        for i, symbol in enumerate(self._symbols):
            code[symbols == symbol] = i
        if (code == illegal_code).any():
            raise AlphabetError("Symbol list contains illegal symbols")
        return code
            
    
    def decode_multiple(self, code):
        """
        Decode a sequence code into a list of symbols.
        
        Parameters
        ----------
        code : ndarray
            The sequence code to decode.
        
        Returns
        -------
        symbols : ndarray, dtype='U1'
            The decoded list of symbols.
        """
        symbols = np.zeros(len(code), dtype="U1")
        try:
            # This is only efficient for short alphabets
            # Therefore it is only used in the LetterAlphabet class
            for i, symbol in enumerate(self._symbols):
                symbols[code == i] = symbol
        except IndexError:
            raise AlphabetError("'{:}' is not a valid code"
                                .format(str(i)))
        return symbols
    
    def __str__(self):
        return str(self._symbols.tolist())
    

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