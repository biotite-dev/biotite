# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.sequence"
__author__ = "Patrick Kunzmann"
__all__ = [
    "Alphabet",
    "LetterAlphabet",
    "AlphabetMapper",
    "AlphabetError",
    "common_alphabet",
]

import string
import warnings
from collections.abc import Iterable, Iterator
from collections.abc import Sequence as SequenceABC
from numbers import Integral
from typing import Any, Generic
import numpy as np
from biotite.rust.sequence import AlphabetCodec
from biotite.typing import K, NDArray1, S


class Alphabet(Generic[S]):
    """
    This class defines the allowed symbols for a :class:`Sequence` and
    handles the encoding/decoding between symbols and symbol codes.

    An :class:`Alphabet` is created with the list of symbols, that can
    be used in this context.
    In most cases a symbol will be simply a letter, hence a string of
    length 1. But in principle every hashable Python object can serve
    as symbol.

    The encoding of a symbol into a symbol code is
    done in the following way: Find the first index in the symbol list,
    where the list element equals the symbol. This index is the
    symbol code. If the symbol is not found in the list, an
    :class:`AlphabetError` is raised.

    Internally, a dictionary is used for encoding, with symbols as keys
    and symbol codes as values. Therefore, every symbol must be
    hashable. For decoding the symbol list is indexed with the symbol
    code.

    If an alphabet *1* contains the same symbols and the same
    symbol-code-mappings like another alphabet *2*, but alphabet *1*
    introduces also new symbols, then alphabet *1* *extends* alphabet
    *2*.
    Per definition, every alphabet also extends itself.

    Objects of this class are immutable.

    Parameters
    ----------
    symbols : iterable object
        The symbols, that are allowed in this alphabet. The
        corresponding code for a symbol, is the index of that symbol
        in this list.

    Notes
    -----
    The class is generic in the type of symbols stored in the alphabet.

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
    ... except Exception as e:
    ...    print(e)
    Symbol 'foo' is not in the alphabet

    Create an Alphabet of arbitrary objects:

    >>> alph = Alphabet(["foo", 42, (1,2,3), 5, 3.141])
    >>> print(alph.encode((1,2,3)))
    2
    >>> print(alph.decode(4))
    3.141

    On the subject of alphabet extension:
    An alphabet always extends itself.

    >>> Alphabet(["A","C","G","T"]).extends(Alphabet(["A","C","G","T"]))
    True

    An alphabet extends an alphabet when it contains additional symbols...

    >>> Alphabet(["A","C","G","T","U"]).extends(Alphabet(["A","C","G","T"]))
    True

    ...but not vice versa

    >>> Alphabet(["A","C","G","T"]).extends(Alphabet(["A","C","G","T","U"]))
    False

    Two alphabets with same symbols but different symbol-code-mappings

    >>> Alphabet(["A","C","G","T"]).extends(Alphabet(["A","C","T","G"]))
    False
    """

    def __init__(self, symbols: Iterable[S]) -> None:
        self._symbols: tuple[S, ...] = tuple(symbols)
        if len(self._symbols) == 0:
            raise ValueError("Symbol list is empty")
        self._symbol_dict = {}
        for i, symbol in enumerate(self._symbols):
            self._symbol_dict[symbol] = i

    def __repr__(self) -> str:
        """Represent Alphabet as a string for debugging."""
        return f"Alphabet({self._symbols})"

    def get_symbols(self) -> tuple[S, ...]:
        """
        Get the symbols in the alphabet.

        Returns
        -------
        symbols : tuple
            The symbols.
        """
        return self._symbols

    def extends(self, alphabet: Alphabet[Any]) -> bool:
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
            return alphabet.get_symbols() == self.get_symbols()[: len(alphabet)]

    def encode(self, symbol: S) -> int:
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
            raise AlphabetError(f"Symbol {repr(symbol)} is not in the alphabet")

    def decode(self, code: int) -> S:
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
        if code < 0 or code >= len(self._symbols):
            raise AlphabetError(f"'{code:d}' is not a valid code")
        return self._symbols[code]

    def encode_multiple(
        self,
        symbols: SequenceABC[S],
        dtype: np.dtype | type | str = np.int64,
    ) -> NDArray1[K, np.integer]:
        """
        Encode a list of symbols.

        Parameters
        ----------
        symbols : array-like
            The symbols to encode.
        dtype : dtype, optional
            The dtype of the output ndarray.

        Returns
        -------
        code : ndarray
            The sequence code.
        """
        return np.array([self.encode(e) for e in symbols], dtype=dtype)

    def decode_multiple(
        self, code: NDArray1[K, np.integer] | Iterable[int]
    ) -> SequenceABC[S]:
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
        if isinstance(code, np.ndarray):
            return [self.decode(c) for c in code.tolist()]
        else:
            return [self.decode(c) for c in code]

    def is_letter_alphabet(self) -> bool:
        """
        Check whether the symbols in this alphabet are single printable
        letters.
        If so, the alphabet could be expressed by a `LetterAlphabet`.

        Returns
        -------
        is_letter_alphabet : bool
            True, if all symbols in the alphabet are 'str' or 'bytes',
            have length 1 and are printable.
        """
        for symbol in self:
            if not isinstance(symbol, (str, bytes)) or len(symbol) > 1:
                return False
            if isinstance(symbol, str):
                symbol = symbol.encode("ASCII")
            if symbol not in LetterAlphabet.PRINTABLES:
                return False
        return True

    def __str__(self) -> str:
        return str(self.get_symbols())

    def __len__(self) -> int:
        return len(self.get_symbols())

    def __iter__(self) -> Iterator[S]:
        return self.get_symbols().__iter__()

    def __contains__(self, symbol: object) -> bool:
        return symbol in self.get_symbols()

    def __hash__(self) -> int:
        symbols = self.get_symbols()
        if isinstance(symbols, tuple):
            return hash(symbols)
        else:
            return hash(tuple(symbols))

    def __eq__(self, item: object) -> bool:
        if item is self:
            return True
        if not isinstance(item, Alphabet):
            return False
        return self.get_symbols() == item.get_symbols()


class LetterAlphabet(Alphabet[str]):
    """
    :class:`LetterAlphabet` is a an :class:`Alphabet` subclass
    specialized for letter based alphabets, like DNA or protein
    sequence alphabets.
    The alphabet size is limited to the 94 printable, non-whitespace
    characters.
    Internally the symbols are saved as `bytes` objects.
    The encoding and decoding process is a lot faster than for a
    normal :class:`Alphabet`.

    The performance gain comes through the use of *NumPy* and *Cython*
    for encoding and decoding, without the need of a dictionary.

    Parameters
    ----------
    symbols : iterable object or str or bytes
        The symbols, that are allowed in this alphabet. The
        corresponding code for a symbol, is the index of that symbol
        in this list.
    """

    PRINTABLES = (string.digits + string.ascii_letters + string.punctuation).encode(
        "ASCII"
    )

    def __init__(self, symbols: Iterable[str | bytes] | str | bytes) -> None:
        symbols_list = list(symbols)
        if len(symbols_list) == 0:
            raise ValueError("Symbol list is empty")
        symbols_bytes = []
        for symbol in symbols_list:
            if not isinstance(symbol, (str, bytes)) or len(symbol) > 1:
                raise ValueError(f"Symbol '{symbol}' is not a single letter")
            if isinstance(symbol, str):
                symbol = symbol.encode("ASCII")
            if symbol not in LetterAlphabet.PRINTABLES:
                raise ValueError(
                    f"Symbol {repr(symbol)} is not printable or whitespace"
                )
            symbols_bytes.append(symbol)
        # Direct 'astype' conversion is not allowed by numpy
        # -> frombuffer()
        self._symbols: NDArray1[Any, np.ubyte] = np.frombuffer(
            np.array(symbols_bytes, dtype="|S1"), dtype=np.ubyte
        )
        self._codec = AlphabetCodec(self._symbols)

    def __repr__(self) -> str:
        """Represent LetterAlphabet as a string for debugging."""
        return f"LetterAlphabet({self.get_symbols()})"

    def extends(self, alphabet: Alphabet[Any]) -> bool:
        if alphabet is self:
            return True
        elif isinstance(alphabet, LetterAlphabet):
            if len(alphabet._symbols) > len(self._symbols):
                return False
            return bool(
                np.all(alphabet._symbols == self._symbols[: len(alphabet._symbols)])
            )
        else:
            return super().extends(alphabet)

    def get_symbols(self) -> tuple[str, ...]:
        return tuple([symbol.decode("ASCII") for symbol in self._symbols_as_bytes()])

    def encode(self, symbol: str | bytes) -> int:
        if not isinstance(symbol, (str, bytes)) or len(symbol) > 1:
            raise AlphabetError(f"Symbol '{symbol}' is not a single letter")
        indices = np.where(self._symbols == ord(symbol))[0]
        if len(indices) == 0:
            raise AlphabetError(f"Symbol {repr(symbol)} is not in the alphabet")
        return indices[0].item()

    def decode(self, code: int, as_bytes: bool = False) -> str:
        if as_bytes:
            warnings.warn(
                "'as_bytes' is deprecated, use 'decode_into_byte()' instead",
                DeprecationWarning,
                stacklevel=2,
            )
        if code < 0 or code >= len(self._symbols):
            raise AlphabetError(f"'{code:d}' is not a valid code")
        return chr(self._symbols[code])

    def decode_into_byte(self, code: int) -> bytes:
        """
        Decode a code into a single-byte symbol.

        Parameters
        ----------
        code : int
            The code to decode.

        Returns
        -------
        symbol : bytes
            The decoded symbol as a single-byte object.
        """
        if code < 0 or code >= len(self._symbols):
            raise AlphabetError(f"'{code:d}' is not a valid code")
        return bytes([int(self._symbols[code])])

    def encode_multiple(
        self,
        symbols: Iterable[str | bytes]
        | str
        | bytes
        | NDArray1[K, np.character]
        | NDArray1[K, np.unsignedinteger],
        dtype: Any = None,
    ) -> NDArray1[K, np.integer]:
        """
        Encode multiple symbols.

        Parameters
        ----------
        symbols : iterable object or str or bytes
            The symbols to encode. The method is fastest when a
            :class:`ndarray`, :class:`str` or :class:`bytes` object
            containing the symbols is provided, instead of e.g. a list.
        dtype : dtype, optional
            For compatibility with superclass. The value is ignored.

        Returns
        -------
        code : ndarray
            The sequence code.
        """
        if isinstance(symbols, str):
            symbols = np.frombuffer(symbols.encode("ASCII"), dtype=np.ubyte)
        elif isinstance(symbols, bytes):
            symbols = np.frombuffer(symbols, dtype=np.ubyte)
        elif isinstance(symbols, np.ndarray):
            symbols = np.frombuffer(symbols.astype(dtype="|S1"), dtype=np.ubyte)
        else:
            symbols = np.frombuffer(
                np.array(list(symbols), dtype="|S1"), dtype=np.ubyte
            )
        return self._codec.encode(symbols)

    def decode_multiple(
        self,
        code: NDArray1[K, np.integer] | Iterable[int],
        as_bytes: bool = False,
    ) -> NDArray1[K, np.character]:
        """
        Decode a sequence code into a list of symbols.

        Parameters
        ----------
        code : ndarray, dtype=uint8
            The sequence code to decode.
            Works fastest if a :class:`ndarray` is provided.
        as_bytes : bool, optional
            DEPRECATED: Use :meth:`decode_multiple_into_bytes()` instead.
            If true, the output array will contain `bytes`
            (dtype 'S1').
            Otherwise, the the output array will contain `str`
            (dtype 'U1').

        Returns
        -------
        symbols : ndarray, dtype='U1' or dtype='S1'
            The decoded list of symbols.
        """
        if as_bytes:
            warnings.warn(
                "'as_bytes' is deprecated, use 'decode_multiple_into_bytes()' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.decode_multiple_into_bytes(code)
        if not isinstance(code, np.ndarray):
            code = np.array(code, dtype=np.uint8)
        code = code.astype(np.uint8, copy=False)
        symbols = self._codec.decode(code)
        # Symbols must be convverted from 'np.ubyte' to '|S1'
        symbols = np.frombuffer(symbols, dtype="|S1")
        return symbols.astype("U1")

    def decode_multiple_into_bytes(
        self,
        code: NDArray1[K, np.integer] | Iterable[int],
    ) -> NDArray1[K, np.bytes_]:
        """
        Decode a sequence code into a single-byte symbol array.

        This is a faster alternative to :meth:`decode_multiple()` when
        single-byte symbols are desired, as it avoids the conversion to
        Unicode characters.

        Parameters
        ----------
        code : ndarray, dtype=uint8
            The sequence code to decode.
            Works fastest if a :class:`ndarray` is provided.

        Returns
        -------
        symbols : ndarray, dtype='S1'
            The decoded list of symbols.
        """
        if not isinstance(code, np.ndarray):
            code = np.array(code, dtype=np.uint8)
        code = code.astype(np.uint8, copy=False)
        symbols = self._codec.decode(code)
        return np.frombuffer(symbols, dtype="|S1")

    def is_letter_alphabet(self) -> bool:
        return True

    def __contains__(self, symbol: object) -> bool:
        if not isinstance(symbol, (str, bytes)):
            return False
        return ord(symbol) in self._symbols

    def __len__(self) -> int:
        return len(self._symbols)

    def _symbols_as_bytes(self) -> NDArray1[Any, np.bytes_]:
        """
        Properly convert from dtype 'np.ubyte' to '|S1'
        """
        return np.frombuffer(self._symbols, dtype="|S1")


class AlphabetMapper(object):
    """
    This class is used for symbol code conversion from a source
    alphabet into a target alphabet.

    This means that the symbol codes are converted from one to another
    alphabet so that the symbol itself is preserved.
    This class works for single symbol codes or an entire sequence code
    likewise.

    Parameters
    ----------
    source_alphabet, target_alphabet : Alphabet
        The codes are converted from the source alphabet into the
        target alphabet.
        The target alphabet must contain at least all symbols of the
        source alphabet, but it is not required that the shared symbols
        are in the same order.

    Examples
    --------

    >>> source_alph = Alphabet(["A","C","G","T"])
    >>> target_alph = Alphabet(["T","U","A","G","C"])
    >>> mapper = AlphabetMapper(source_alph, target_alph)
    >>> print(mapper[0])
    2
    >>> print(mapper[1])
    4
    >>> print(mapper[[1,1,3]])
    [4 4 0]
    >>> in_sequence = GeneralSequence(source_alph, "GCCTAT")
    >>> print(in_sequence.code)
    [2 1 1 3 0 3]
    >>> print("".join(in_sequence.symbols))
    GCCTAT
    >>> out_sequence = GeneralSequence(target_alph)
    >>> out_sequence.code = mapper[in_sequence.code]
    >>> print(out_sequence.code)
    [3 4 4 0 2 0]
    >>> print("".join(out_sequence.symbols))
    GCCTAT
    """

    def __init__(
        self, source_alphabet: Alphabet[Any], target_alphabet: Alphabet[Any]
    ) -> None:
        if target_alphabet.extends(source_alphabet):
            self._necessary_mapping = False
        else:
            self._necessary_mapping = True
            self._mapper = np.zeros(
                len(source_alphabet), dtype=AlphabetMapper._dtype(len(target_alphabet))
            )
            for old_code in range(len(source_alphabet)):
                symbol = source_alphabet.decode(old_code)
                new_code = target_alphabet.encode(symbol)
                self._mapper[old_code] = new_code

    def __getitem__(
        self, code: int | NDArray1[K, np.integer]
    ) -> int | NDArray1[K, np.integer]:
        if isinstance(code, Integral):
            if self._necessary_mapping:
                return self._mapper[code]
            else:
                return code
        if not isinstance(code, np.ndarray) or code.dtype not in (
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ):
            code = np.array(code, dtype=np.uint64)
        if self._necessary_mapping:
            return self._mapper[code]
        else:
            return code

    @staticmethod
    def _dtype(alphabet_size: int) -> type[np.unsignedinteger]:
        _size_uint8 = np.iinfo(np.uint8).max + 1
        _size_uint16 = np.iinfo(np.uint16).max + 1
        _size_uint32 = np.iinfo(np.uint32).max + 1
        if alphabet_size <= _size_uint8:
            return np.uint8
        elif alphabet_size <= _size_uint16:
            return np.uint16
        elif alphabet_size <= _size_uint32:
            return np.uint32
        else:
            return np.uint64


class AlphabetError(Exception):
    """
    This exception is raised, when a code or a symbol is not in an
    :class:`Alphabet`.
    """

    pass


def common_alphabet(
    alphabets: Iterable[Alphabet[S]],
) -> Alphabet[S] | None:
    """
    Determine the alphabet from a list of alphabets, that
    extends all alphabets.

    Parameters
    ----------
    alphabets : iterable of Alphabet
        The alphabets from which the common one should be identified.

    Returns
    -------
    common_alphabet : Alphabet or None
        The alphabet from `alphabets` that extends all alphabets.
        ``None`` if no such common alphabet exists.
    """
    common_alphabet = None
    for alphabet in alphabets:
        if common_alphabet is None:
            common_alphabet = alphabet
        elif not common_alphabet.extends(alphabet):
            if alphabet.extends(common_alphabet):
                common_alphabet = alphabet
            else:
                return None
    return common_alphabet
