# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Optional, Union, TextIO, BinaryIO
from .atoms import AtomArrayStack, AtomArray


def load_structure(
    file_path: str,
    template: Optional[Union[AtomArrayStack, AtomArray]] = None
) -> Union[AtomArray, AtomArrayStack, TextIO, BinaryIO, str]: ...

def save_structure(file_path: str, array: AtomArrayStack) -> None: ...
