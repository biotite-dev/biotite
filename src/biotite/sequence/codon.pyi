# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Dict, Tuple, Union, Iterable, List, overload


class CodonTable:
    def __init__(
        self, codon_dict: Dict[str, str], starts: Iterable[str]
    ) -> None: ...
    # Codon or amino acid as str
    @overload
    def __getitem__(
        self, index: str
    ) -> Union[str, Tuple[str, ...]]: ...
    # Codon as code tuple
    @overload
    def __getitem__(
        self, index: Tuple[int, int, int]
    ) -> int: ...
    # Amino acid as code
    @overload
    def __getitem__(
        self, index: int
    ) -> Tuple[Tuple[int, int, int], ...]: ...
    def __str__(self) -> str: ...
    @overload
    def codon_dict(self) -> Dict[str, str]: ...
    @overload
    def codon_dict(
        self, code: bool = False
    ) -> Union[
        Dict[str, str],
        Dict[Tuple[int, int, int], int]
    ]: ...
    @overload
    def start_codons(self) -> Tuple["str"]: ...
    @overload
    def start_codons(
        self, code: bool = False
    ) -> Union[Tuple["str"], Tuple[Tuple[int, int, int]]]: ...
    @staticmethod
    def load(table_name: Union[str, int]) -> CodonTable: ...
    @staticmethod
    def table_names() -> List[str]: ...
    @staticmethod
    def default_table() -> CodonTable: ...
