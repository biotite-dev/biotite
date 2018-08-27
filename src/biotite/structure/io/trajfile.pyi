# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Optional, Union, BinaryIO
import numpy as np
from ..atoms import AtomArray, AtomArrayStack
from ...file import File


class TrajectoryFile(File[BinaryIO]):
    def __init__(self) -> None: ...
    def read(
        self,
        file_name: str,
        start: Optional[int] = None,
        stop:  Optional[int] = None,
        step:  Optional[int] = None,
        atom_i: Optional[np.ndarray] = None
    ) -> None: ...
    def get_coord(self) -> np.ndarray: ...
    def get_structure(
        self,
        template: Union[AtomArray, AtomArrayStack]
    ) -> AtomArrayStack: ...
    def get_time(self) -> np.ndarray: ...
    def get_box(self) -> np.ndarray: ...
    def write(self, file_name: str) -> None: ...
