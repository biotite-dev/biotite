from numpy import (
    int64,
    uint8,
)
from typing import (
    Dict,
    List,
    Tuple,
    Union,
)


class CodonTable:
    def __getitem__(
        self,
        item: Union[Tuple[int, int, int], Tuple[uint8, uint8, uint8], str, int]
    ) -> Union[int64, str, Tuple[Tuple[uint8, uint8, uint8], Tuple[uint8, uint8, uint8]], Tuple[str, str]]: ...
    def __init__(self, codon_dict: Dict[str, str], starts: List[str]) -> None: ...
    def codon_dict(self, code: bool = False) -> Dict[str, str]: ...
    @staticmethod
    def default_table() -> CodonTable: ...
    @staticmethod
    def load(table_name: Union[str, int]) -> CodonTable: ...
    def start_codons(self, code: bool = False) -> Tuple[Tuple[uint8, uint8, uint8]]: ...
