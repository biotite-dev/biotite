from collections import OrderedDict
from numpy import (
    ndarray,
    str_,
)
from typing import (
    Dict,
    List,
    Optional,
    Union,
)


def _data_block_name(line: str) -> Optional[str]: ...


def _get_category_name(line: str) -> Optional[str]: ...


def _is_empty(line: str) -> bool: ...


def _is_loop_start(line: str) -> bool: ...


def _is_multi(line: str, is_loop: bool) -> bool: ...


def _process_looped(lines: List[str], whitepace_values: bool) -> Dict[str, ndarray]: ...


def _process_singlevalued(lines: List[str]) -> Dict[str, str]: ...


def _quote(value: str_) -> str: ...


class PDBxFile:
    def __getitem__(self, index: str) -> Dict[str, Union[str, ndarray]]: ...
    def __init__(self) -> None: ...
    def _add_category(
        self,
        block: str,
        category_name: Optional[str],
        start: int,
        stop: int,
        is_loop: bool,
        is_multilined: bool
    ) -> None: ...
    def get_block_names(self) -> List[str]: ...
    def get_category(self, category: str, block: None = None) -> Dict[str, Union[str, ndarray]]: ...
    def read(self, file: str) -> None: ...
    def set_category(self, category: str, category_dict: OrderedDict, block: Optional[str] = None) -> None: ...
