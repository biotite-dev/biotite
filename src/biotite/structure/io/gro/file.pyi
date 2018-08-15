# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, overload
from ...atoms import AtomArray, AtomArrayStack
from ....file import TextFile


class GROFile(TextFile):
    @overload
    def get_structure(
        self, model: None = None
    ) -> AtomArrayStack: ...
    @overload
    def get_structure(
        self, model: int
    ) -> AtomArray: ...
    def set_structure(
        self, array: Union[AtomArray, AtomArrayStack]
    ) -> None: ...
