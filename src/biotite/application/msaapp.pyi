# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Optional, Iterable, List, Type, TypeVar
import numpy as np
from abc import abstractmethod
from .localapp import LocalApp
from ...sequence.sequence import Sequence
from ...sequence.align.alignment import Alignment


_T = TypeVar('_T', bound=MSAApp)

class MSAApp(LocalApp):
    def __init__(
        self,
        sequences: Iterable[Sequence],
        bin_path: None = None,
        mute: bool = True
    ) -> None: ...
    def run(self) -> None: ...
    def evaluate(self) -> None: ...
    def get_alignment(self) -> Alignment: ...
    def get_alignment_order(self) -> np.ndarray: ...
    @abstractmethod
    def get_default_bin_path(self) -> str: ...
    @abstractmethod
    def get_cli_arguments(self) -> List[str]: ...
    def get_input_file_path(self) -> str: ...
    def get_output_file_path(self) -> str: ...
    @classmethod
    def align(
        cls: Type[_T], sequences: Iterable[Sequence], bin_path: Optional[str] = None
    ) -> Alignment: ...
