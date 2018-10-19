# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Optional, List, Dict, NewType, Callable
from ..annotation import Feature, Annotation
try:
    from matplotlib.axes import Axes
    from matplotlib.transforms import Bbox
except ImportError:
    Axes = NewType("Axes", object)
    Bbox = NewType("Bbox", object)


def plot_feature_map(
    axes: Axes,
    annotation: Annotation,
    loc_range = Tuple[int,int],
    multi_line: bool = True,
    symbols_per_line: int = 1000,
    show_numbers: bool = False,
    number_size: Optional[float] = None,
    line_width: float = 0.05,
    show_line_position: bool = False,
    spacing: float = 0.25,
    feature_plotters: Optional[FeaturePlotter] = None,
    style_param: Optional[Dict[str, object]] = None
) -> None: ...

class FeaturePlotter():
    def __init__(self) -> None: ...
    def matches(self, feature: Feature) -> bool: ...
    def draw(
        self,
        axes: Axes,
        feature: Feature,
        bbox: BBox,
        loc_index: int,
        style_param: Optional[Dict[str, object]]
    ) -> None: ...

class CodingPlotter(FeaturePlotter):
    def __init__(self,
        tail_width: float = 0.5,
        head_width: float = 0.8
    ) -> None: ...

class MiscFeaturePlotter(FeaturePlotter):
    def __init__(self, height: float = 0.4) -> None: ...

class PromoterPlotter(FeaturePlotter):
    def __init__(
        self,
        line_width: float = 2,
        head_width: float = 2,
        head_length: float = 6,
        head_height: float = 0.8
    ) -> None: ...

class TerminatorPlotter(FeaturePlotter):
    def __init__(self, bar_width: float = 5) -> None: ...

class RBSPlotter(FeaturePlotter):
    def __init__(self, height: float = 0.4) -> None: ...