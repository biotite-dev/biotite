# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Optional, Union
from ...atoms import AtomArray, AtomArrayStack
from ....file import TextFile


class GROFile(TextFile):
    def get_structure(
        self, model: Optional[int] = None
    ) -> Union[AtomArray, AtomArrayStack]: ...
    def set_structure(
        self, array: Union[AtomArray, AtomArrayStack]
    ) -> None: ...
