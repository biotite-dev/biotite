# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Optional, List, Any
from ..phylo.tree import Tree
try:
    from matplotlib.axes import Axes
except ImportError:
    Axes = Any


def plot_dendrogram(
    axes: Axes,
    tree: Tree,
    orientation: str = "left",
    use_distances: bool = True,
    labels: Optional[List[str]] = None,
    label_size: Optional[float] = None,
    color: str = "black",
    show_distance: bool = True,
    **kwargs
) -> None: ...