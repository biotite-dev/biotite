# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, TypeVar


_T = TypeVar("_T", bound="Copyable")

class Copyable:
    def __init__(self) -> None: ...
    def __copy_create__(self: _T) -> _T: ...
    def __copy_fill__(self: _T, clone: _T) -> None: ...
    def copy(self: _T) -> _T: ...
