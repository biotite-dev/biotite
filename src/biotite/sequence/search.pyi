from biotite.sequence.seqtypes import NucleotideSequence
from numpy import (
    int64,
    ndarray,
)


def find_subsequence(
    sequence: NucleotideSequence,
    query: NucleotideSequence
) -> ndarray: ...


def find_symbol(sequence: NucleotideSequence, symbol: str) -> ndarray: ...


def find_symbol_first(sequence: NucleotideSequence, symbol: str) -> int64: ...


def find_symbol_last(sequence: NucleotideSequence, symbol: str) -> int64: ...
