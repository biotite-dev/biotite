# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Optional, List, NewType
from ..align.alignment import Alignment
from .colorschemes import Color
try:
    from matplotlib.axes import Axes
except ImportError:
    Axes = NewType("Axes", object)


def plot_sequence_logo(
    axes: Axes,
    alignment: Alignment,
    scheme: Optional[List[Color]] = None,
    **kwargs
) -> None: