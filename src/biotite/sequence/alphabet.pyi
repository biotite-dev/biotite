from numpy import (
    int64,
    ndarray,
    str_,
    uint8,
)
from typing import (
    Iterator,
    List,
    Optional,
    Type,
    Union,
)


class Alphabet:
    def __contains__(self, symbol: str) -> bool: ...
    def __init__(self, symbols: str) -> None: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def decode(self, code: Union[int64, uint8, int]) -> str_: ...
    def extends(self, alphabet: Alphabet) -> bool: ...


class LetterAlphabet:
    def __init__(self, symbols: List[str]) -> None: ...
    def encode(self, symbol: str) -> int64: ...
    def encode_multiple(
        self,
        symbols: Union[List[str], str],
        dtype: Optional[Type[uint8]] = None
    ) -> ndarray: ...
