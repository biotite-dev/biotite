from biotite.sequence.alphabet import (
    Alphabet,
    LetterAlphabet,
)
from numpy import (
    int64,
    ndarray,
)
from typing import (
    Dict,
    List,
    Tuple,
    Union,
)


class SubstitutionMatrix:
    def __init__(
        self,
        alphabet1: Alphabet,
        alphabet2: Alphabet,
        score_matrix: Union[str, ndarray]
    ) -> None: ...
    def __str__(self) -> str: ...
    def _fill_with_matrix_dict(self, matrix_dict: Dict[Tuple[str, str], int64]) -> None: ...
    @staticmethod
    def dict_from_db(matrix_name: str): ...
    @staticmethod
    def dict_from_str(string: str) -> Dict[Tuple[str, str], int64]: ...
    def get_alphabet1(self) -> LetterAlphabet: ...
    def get_alphabet2(self) -> LetterAlphabet: ...
    @staticmethod
    def list_db() -> List[str]: ...
    def score_matrix(self) -> ndarray: ...
    @staticmethod
    def std_nucleotide_matrix() -> SubstitutionMatrix: ...
