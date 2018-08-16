# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Dict, Tuple
from abc import abstractmethod
from matplotlib.figure import Figure

class Visualizer():
    def __init__(self) -> None: ...
    def create_figure(
        self, size: Tuple[int, int], dpi: int = 100
    ) -> Figure: ...
    @abstractmethod
    def generate(self) -> Figure: ...

colors: Dict[str, str]