# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Optional, Union, Tuple, List, overload
from ...atoms import AtomArray, AtomArrayStack
from .file import PDBFile
from ....sequence.seqtypes import ProteinSequence

@overload
def get_structure(
    pdb_file: PDBFile,
    model: None = None,
    insertion_code: List[Tuple[int, str]] = [],
    altloc: List[Tuple[int, str]] = [],
    extra_fields: List[str] = []
) -> AtomArrayStack: ...
@overload
def get_structure(
    pdb_file: PDBFile,
    model: int,
    insertion_code: List[Tuple[int, str]] = [],
    altloc: List[Tuple[int, str]] = [],
    extra_fields: List[str] = []
) -> AtomArray: ...

def set_structure(
    pdb_file: PDBFile,
    array: Union[AtomArray, AtomArrayStack],
) -> None: ...
