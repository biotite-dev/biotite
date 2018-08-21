# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Optional
import numpy as np
from .localapp import LocalApp
from ...structure.atoms import AtomArray


class DsspApp(LocalApp):
    def __init__(
        self, atom_array: AtomArray, bin_path: str = "dssp", mute: bool = True
    ) -> None: ...
    def run(self) -> None: ...
    def evaluate(self) -> None: ...
    def get_sse(self) -> np.ndarray: ...
    @staticmethod
    def annotate_sse(
        atom_array: AtomArray, bin_path: str = "dssp"
    ) -> np.ndarray: ...
