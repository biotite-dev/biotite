from biotite.structure.atoms import (
    AtomArray,
    AtomArrayStack,
)
from typing import (
    Optional,
    Union,
)


class GROFile:
    def get_structure(
        self,
        model: Optional[int] = None
    ) -> Union[AtomArray, AtomArrayStack]: ...
    def set_structure(
        self,
        array: Union[AtomArray, AtomArrayStack]
    ) -> None: ...
