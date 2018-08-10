from biotite.structure.atoms import (
    AtomArray,
    AtomArrayStack,
)
from biotite.structure.io.pdbx.file import PDBxFile
from numpy import ndarray
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)


def _determine_entity_id(chain_id: ndarray) -> ndarray: ...


def _fill_annotations(
    array: Union[AtomArray, AtomArrayStack],
    model_dict: Dict[str, ndarray],
    extra_fields: List[str]
) -> None: ...


def _filter_inscode_altloc(
    array: Union[AtomArray, AtomArrayStack],
    model_dict: Dict[str, ndarray],
    inscode: List[Any],
    altloc: List[Any]
) -> Union[AtomArray, AtomArrayStack]: ...


def _get_model_dict(atom_site_dict: Dict[str, ndarray], model: int) -> Dict[str, ndarray]: ...


def get_structure(
    pdbx_file: PDBxFile,
    data_block: None = None,
    insertion_code: List[Any] = [],
    altloc: List[Any] = [],
    model: Optional[int] = None,
    extra_fields: List[str] = []
) -> Union[AtomArray, AtomArrayStack]: ...


def set_structure(
    pdbx_file: PDBxFile,
    array: Union[AtomArray, AtomArrayStack],
    data_block: Optional[str] = None
) -> None: ...
