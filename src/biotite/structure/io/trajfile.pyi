from biotite.structure.atoms import (
    AtomArray,
    AtomArrayStack,
)
from typing import Union


class TrajectoryFile:
    def __init__(self) -> None: ...
    def get_structure(
        self,
        template: Union[AtomArray, AtomArrayStack]
    ) -> AtomArrayStack: ...
    def read(
        self,
        file_name: str,
        start: None = None,
        stop: None = None,
        step: None = None,
        atom_i: None = None
    ) -> None: ...
