# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Optional, List, Union, Tuple, Any
from ..align.alignment import Alignment
try:
    from matplotlib.axes import Axes
except ImportError:
    Axes = Any


Color = Union[str, Tuple[float, float, float]]


def plot_sequence_logo(
    axes: Axes,
    alignment: Alignment,
    scheme: Optional[List[Color]] = None,
    **kwargs
) -> None: ...