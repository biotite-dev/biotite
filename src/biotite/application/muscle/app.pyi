# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Iterable, Optional, List, Union, Tuple
from ..msaapp import MSAApp
from ...sequence.seqtypes import ProteinSequence
from ...sequence.align.matrix import SubstitutionMatrix


class MuscleApp:
    def __init__(
        self,
        sequences: Iterable[ProteinSequence],
        bin_path: Optional[str] = None,
        mute: bool = True
    ) -> None: ...
    def set_matrix(self, matrix: SubstitutionMatrix) -> None: ...
    def set_gap_penalty(
        self,
        gap_penalty: Union[float, Tuple[float,float]]
    ) -> None: ...
    def get_cli_arguments(self) -> List[str]: ...
    @staticmethod
    def get_default_bin_path() -> str: ...
