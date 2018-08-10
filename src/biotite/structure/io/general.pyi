from biotite.structure.atoms import (
    AtomArray,
    AtomArrayStack,
)
from typing import (
    Optional,
    Union,
)


def load_structure(
    file_path: str,
    template: Optional[AtomArrayStack] = None
) -> Union[AtomArray, AtomArrayStack]: ...


def save_structure(file_path: str, array: AtomArrayStack) -> None: ...
