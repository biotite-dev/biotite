# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Optional, Tuple, Union, Iterable, List, Hashable
import numpy as np
from ..sequence import Sequence
from .matrix import SubstitutionMatrix


class Alignment:
    sequences : List[Sequence]
    trace : np.ndarray
    score: Optional[int]
    def __init__(
        self, sequences: List[Sequence],
        trace: np.ndarray,
        score: Optional[int]
    ) -> None: ...
    def get_gapped_sequences(self) -> List[str]: ...
    def __str__(self) -> str: ...
    def __getitem__(self, index: Union[slice, Tuple]) -> Alignment: ...
    @staticmethod
    def trace_from_strings(seq_str_list: List[str]) -> np.ndarray: ...


def get_codes(alignment: Alignment) -> np.ndarray: ...

def get_symbols(alignment: Alignment) -> List[List[Hashable]]: ...

def get_sequence_identity(
    alignment: Alignment, mode: str = 'not_terminal'
) -> float: ...

def score(
    alignment: Alignment,
    matrix: SubstitutionMatrix,
    gap_penalty: Union[Tuple[int, int], int] = -10,
    terminal_penalty: bool = True
) -> int: ...