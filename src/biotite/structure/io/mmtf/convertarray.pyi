# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union
from ...atoms import AtomArray, AtomArrayStack
from .file import MMTFFile


def set_structure(
    file: MMTFFile,
    array: Union[AtomArray, AtomArrayStack],
) -> None: ...