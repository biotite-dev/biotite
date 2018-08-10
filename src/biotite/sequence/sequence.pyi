from biotite.sequence.seqtypes import (
    NucleotideSequence,
    ProteinSequence,
)
from numpy import (
    bool_,
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


class Sequence:
    def __add__(
        self,
        sequence: NucleotideSequence
    ) -> NucleotideSequence: ...
    def __eq__(
        self,
        item: Union[ProteinSequence, NucleotideSequence]
    ) -> bool: ...
    def __getitem__(
        self,
        index: Union[int64, slice, int]
    ) -> Union[str_, NucleotideSequence]: ...
    def __init__(self, sequence: List[str] = []) -> None: ...
    def __iter__(self) -> Iterator[str_]: ...
    def __len__(self) -> int: ...
    def __setitem__(
        self,
        index: Union[slice, ndarray, int],
        item: Union[str, ndarray, NucleotideSequence]
    ) -> None: ...
    def __str__(self) -> str: ...
    @staticmethod
    def _dtype(alphabet_size: int) -> Type[uint8]: ...
    def copy(
        self,
        new_seq_code: Optional[ndarray] = None
    ) -> Union[ProteinSequence, NucleotideSequence]: ...
    def is_valid(self) -> bool_: ...
    def reverse(self) -> NucleotideSequence: ...
