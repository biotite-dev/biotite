# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, BinaryIO
import numpy as np
from ....file import File
from ...atoms import AtomArrayStack, AtomArray


class NpzFile(File):
    def __init__(self) -> None: ...
    def read(self, file: Union[str, BinaryIO]) -> None: ...
    def write(self, file: Union[str, BinaryIO]) -> None: ...
    def get_structure(self) -> Union[AtomArrayStack, AtomArray]: ...
    def set_structure(
        self, array: Union[AtomArrayStack, AtomArray]
    ) -> None: ...
