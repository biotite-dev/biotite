# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Optional, Union, Tuple, List, overload
from ...atoms import AtomArray, AtomArrayStack
from .file import PDBxFile
from ....sequence.seqtypes import ProteinSequence

def get_sequence(
    pdbx_file: PDBxFile, data_block: Optional[str] = None
) -> ProteinSequence: ...

@overload
def get_structure(
    pdbx_file: PDBxFile,
    model: None = None,
    data_block: Optional[str] = None,
    insertion_code: List[Tuple[int, str]] = [],
    altloc: List[Tuple[int, str]] = [],
    extra_fields: List[str] = []
) -> AtomArrayStack: ...
@overload
def get_structure(
    pdbx_file: PDBxFile,
    model: int,
    data_block: Optional[str] = None,
    insertion_code: List[Tuple[int, str]] = [],
    altloc: List[Tuple[int, str]] = [],
    extra_fields: List[str] = []
) -> AtomArray: ...

def set_structure(
    pdbx_file: PDBxFile,
    array: Union[AtomArray, AtomArrayStack],
    data_block: Optional[str] = None
) -> None: ...
