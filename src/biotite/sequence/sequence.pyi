# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import (
    TypeVar,
    Union,
    Optional,
    Hashable,
    Dict,
    List,
    MutableSequence,
    Iterable,
    Iterator,
    overload
)
from typing import Sequence as _Sequence
from abc import abstractmethod
import numpy as np
from ..copyable import Copyable
from .alphabet import Alphabet


_T = TypeVar("_T", bound="Sequence")

class Sequence(Copyable, _Sequence[Hashable]):
    symbols : List[Hashable]
    code : np.ndarray
    def __init__(self, sequence: Iterable[Hashable] = ()) -> None: ...
    def copy(self: _T, new_seq_code: Optional[np.ndarray] = None) -> _T: ...
    @abstractmethod
    def get_alphabet(self) -> Alphabet: ...
    def reverse(self: _T) -> _T: ...
    def is_valid(self) -> bool: ...
    def get_symbol_frequency(self) -> Dict[Hashable, int]: ...
    @overload
    def __getitem__(self, index: int) -> Hashable: ...
    @overload
    def __getitem__(
        self: _T,
        index: Union[MutableSequence[int], MutableSequence[bool], slice]
    ) -> _T: ...
    @overload
    def __setitem__(self, index: int, item: Hashable) -> None: ...
    @overload
    def __setitem__(
        self,
        index: Union[slice, MutableSequence[int], MutableSequence[bool]],
        item: Union[Iterable[Hashable], np.ndarray]
    ) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Hashable]: ...
    def __eq__(self, item: object) -> bool: ...
    def __str__(self) -> str: ...
    def __add__(self: _T, sequence: _T) -> _T: ...
