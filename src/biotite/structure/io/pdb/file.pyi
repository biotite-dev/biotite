from biotite.structure.atoms import (
    AtomArray,
    AtomArrayStack,
)
from typing import (
    Any,
    List,
    Optional,
    Union,
)


class PDBFile:
    def get_structure(
        self,
        insertion_code: List[Any] = [],
        altloc: List[Any] = [],
        model: Optional[int] = None,
        extra_fields: List[str] = []
    ) -> Union[AtomArrayStack, AtomArray]: ...
    def set_structure(
        self,
        array: Union[AtomArrayStack, AtomArray]
    ) -> None: ...
