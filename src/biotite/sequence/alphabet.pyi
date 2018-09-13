# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import (
    Generic,
    TypeVar,
    Hashable,
    Sized,
    Sequence,
    Iterable,
    Container,
    Iterator,
    List,
    Type,
    Union,
)
import numpy as np


_T = TypeVar("_T", bound=Hashable)

class Alphabet(Generic[_T], Iterable[_T], Container[_T],
               Sized, Hashable):
    def __init__(self, symbols: Iterable[_T]) -> None: ...
    def get_symbols(self) -> Sequence[_T]: ...
    def extends(self, alphabet: Alphabet) -> bool: ...
    def encode(self, symbol: _T) -> int: ...
    def decode(self, code: int) -> _T: ...
    def encode_multiple(
        self,
        symbols: Iterable[_T],
        dtype: Union[Type, np.dtype, None] = None
    ) -> np.ndarray: ...
    def decode_multiple(
        self,
        code: np.ndarray
    ) -> Sequence[_T]: ...
    def __str__(self) -> str: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[_T]: ...
    def __contains__(self, symbol: object) -> bool: ...
    def __hash__(self) -> int: ...


class LetterAlphabet(Alphabet[str]):
    def __init__(self, symbols: List[str]) -> None: ...
    def get_symbols(self) -> List[str]: ...
    def encode(self, symbol: str) -> int: ...
    def decode(self, code: int) -> str: ...
    def encode_multiple(
        self,
        symbols: Iterable[str],
        dtype: Union[Type, np.dtype, None] = None
    ) -> np.ndarray: ...
    def decode_multiple(
        self,
        code: np.ndarray
    ) -> np.ndarray: ...


class AlphabetMapper:
    def __init__(
        self, source_alphabet: Alphabet, target_alphabet: Alphabet
    ) -> None: ...  
    def __getitem__(self, code: int) -> int: ...


class AlphabetError(Exception):
    ...