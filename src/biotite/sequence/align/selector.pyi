__all__ = [
    "MinimizerSelector",
    "SyncmerSelector",
    "CachedSyncmerSelector",
    "MincodeSelector",
]

from typing import Any
import numpy as np
from biotite.sequence.align.kmeralphabet import KmerAlphabet
from biotite.sequence.align.permutation import Permutation
from biotite.sequence.alphabet import Alphabet
from biotite.sequence.sequence import Sequence
from biotite.typing import K, N, NDArray1

class MinimizerSelector:
    def __init__(
        self,
        kmer_alphabet: KmerAlphabet[Any],
        window: int,
        permutation: Permutation | None = None,
    ) -> None: ...
    @property
    def kmer_alphabet(self) -> KmerAlphabet[Any]: ...
    @property
    def window(self) -> int: ...
    @property
    def permutation(self) -> Permutation | None: ...
    def select(
        self, sequence: Sequence[Any], alphabet_check: bool = True
    ) -> tuple[NDArray1[K, np.uint32], NDArray1[K, np.int64]]: ...
    def select_from_kmers(
        self, kmers: NDArray1[N, np.int64]
    ) -> tuple[NDArray1[K, np.uint32], NDArray1[K, np.int64]]: ...

class SyncmerSelector:
    def __init__(
        self,
        alphabet: Alphabet[Any],
        k: int,
        s: int,
        permutation: Permutation | None = None,
        offset: tuple[int, ...] = (0,),
    ) -> None: ...
    @property
    def alphabet(self) -> Alphabet[Any]: ...
    @property
    def kmer_alphabet(self) -> KmerAlphabet[Any]: ...
    @property
    def smer_alphabet(self) -> KmerAlphabet[Any]: ...
    @property
    def permutation(self) -> Permutation | None: ...
    def select(
        self, sequence: Sequence[Any], alphabet_check: bool = True
    ) -> tuple[NDArray1[K, np.uint32], NDArray1[K, np.int64]]: ...
    def select_from_kmers(
        self, kmers: NDArray1[N, np.int64]
    ) -> tuple[NDArray1[K, np.uint32], NDArray1[K, np.int64]]: ...

class CachedSyncmerSelector(SyncmerSelector):
    def __init__(
        self,
        alphabet: Alphabet[Any],
        k: int,
        s: int,
        permutation: Permutation | None = None,
        offset: tuple[int, ...] = (0,),
    ) -> None: ...
    def select(
        self, sequence: Sequence[Any], alphabet_check: bool = True
    ) -> tuple[NDArray1[K, np.uint32], NDArray1[K, np.int64]]: ...
    def select_from_kmers(
        self, kmers: NDArray1[N, np.int64]
    ) -> tuple[NDArray1[K, np.uint32], NDArray1[K, np.int64]]: ...

class MincodeSelector:
    def __init__(
        self,
        kmer_alphabet: KmerAlphabet[Any],
        compression: float,
        permutation: Permutation | None = None,
    ) -> None: ...
    @property
    def kmer_alphabet(self) -> KmerAlphabet[Any]: ...
    @property
    def compression(self) -> float: ...
    @property
    def threshold(self) -> int: ...
    @property
    def permutation(self) -> Permutation | None: ...
    def select(
        self, sequence: Sequence[Any], alphabet_check: bool = True
    ) -> tuple[NDArray1[K, np.uint32], NDArray1[K, np.int64]]: ...
    def select_from_kmers(
        self, kmers: NDArray1[N, np.int64]
    ) -> tuple[NDArray1[K, np.uint32], NDArray1[K, np.int64]]: ...
