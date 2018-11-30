# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import (
    TypeVar,
    Tuple,
    Union,
    Optional,
    Hashable,
    Iterable,
    Dict,
    List,
    overload
)
from abc import abstractmethod
from .sequence import Sequence
from .alphabet import Alphabet, LetterAlphabet
from .codon import CodonTable


_T = TypeVar("_T", bound=Hashable)

class GeneralSequence(Sequence[_T]):
    def __init__(
        self, alphabet : Alphabet[_T], sequence: Iterable[_T] = ()
    ) -> None: ...
    def get_alphabet(self) -> Alphabet[_T]: ...


class NucleotideSequence(Sequence[str]):
    alphabet : LetterAlphabet
    def __init__(
        self, sequence: Iterable[str] = (), ambiguous: bool = False
    ) -> None: ...
    def get_alphabet(self) -> LetterAlphabet: ...
    def complement(self) -> NucleotideSequence: ...
    @overload
    def translate(
        self,
        codon_table: Optional[CodonTable ] = None,
        met_start: bool = False
    ) -> Tuple[List[ProteinSequence], List[Tuple[int, int]]]: ...
    @overload
    def translate(
        self,
        complete: bool = False,
        codon_table: Optional[CodonTable ] = None,
        met_start: bool = False
    ) -> Union[ProteinSequence,
               Tuple[List[ProteinSequence], List[Tuple[int, int]]]]: ...
    @staticmethod
    def unambiguous_alphabet() -> LetterAlphabet: ...
    @staticmethod
    def ambiguous_alphabet() -> LetterAlphabet: ...


class ProteinSequence(Sequence[str]):
    alphabet : LetterAlphabet
    def __init__(self, sequence: Iterable[str] = ()) -> None: ...
    def get_alphabet(self) -> LetterAlphabet: ...
    def remove_stops(self) -> ProteinSequence: ...
    @staticmethod
    def convert_letter_1to3(symbol: str) -> str: ...
    @staticmethod
    def convert_letter_3to1(symbol: str) -> str: ...
