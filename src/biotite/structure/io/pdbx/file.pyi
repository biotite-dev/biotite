# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Optional, Union, Dict, TextIO, List
import numpy as np
from ....file import TextFile


class PDBxFile(TextFile):
    def __init__(self) -> None: ...
    def read(self, file: Union[str, TextIO]) -> None: ...
    def get_block_names(self) -> List[str]: ...
    def get_category(
        self, category: str, block: Optional[str] = None
    ) -> Dict[str, Union[str, np.ndarray]]: ...
    def set_category(
        self,
        category: str,
        category_dict: Dict[str, Union[str, np.ndarray]],
        block: Optional[str] = None
    ) -> None: ...
    def __setitem__(
        self, index: str, item: Dict[str, Union[str, np.ndarray]]
    ) -> None: ...
    def __getitem__(self, index: str) -> Dict[str, Union[str, np.ndarray]]: ...
