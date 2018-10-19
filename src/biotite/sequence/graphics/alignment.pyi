# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Optional, List, Dict, NewType, Callable
from ..align.alignment import Alignment
from ..align.matrix import SubstitutionMatrix
from .colorschemes import Color
try:
    from matplotlib.axes import Axes
    from matplotlib.transforms import Bbox
    from matplotlib.colors import Colormap
except ImportError:
    Axes = NewType("Axes", object)
    Bbox = NewType("Bbox", object)
    Colormap = NewType("Colormap", object)


class SymbolPlotter():
    axes: Axes
    def __init__(self, axes: Axes) -> None: ...
    def plot_symbol(
        self, bbox: Bbox, alignment: Alignment, column_i: int, seq_i: int
    ) -> None: ...

class LetterPlotter(SymbolPlotter):
    def __init__(
        self,
        axes: Axes,
        color_symbols: bool = False, 
        font_size: Optional[float] = None, 
        font_param: Optional[Dict[str, object]] = None
    ) -> None: ...
    def get_color(
        self,
        alignment: Alignment,
        column_i: int,
        seq_i: int
    ) -> Color: ...

class LetterSimilarityPlotter(LetterPlotter):
    def __init__(
        self,
        axes: Axes,
        matrix: Optional[SubstitutionMatrix] = None,
        color_symbols: bool = False, 
        font_size: Optional[float] = None, 
        font_param: Optional[Dict[str, object]] = None
    ) -> None: ...
    def set_color(self,
        color: Optional[Color] = None,
        cmap: Optional[Colormap] = None
    ) -> None: ...

class LetterTypePlotter(LetterPlotter):
    def __init__(
        self,
        axes: Axes,
        color_scheme: Optional[List[Color]] = None,
        color_symbols: bool = False, 
        font_size: Optional[float] = None, 
        font_param: Optional[Dict[str, object]] = None
    ) -> None: ...


def plot_alignment(
    axes: Axes,
    alignment: Alignment,
    symbol_plotter: SymbolPlotter,
    symbols_per_line: int = 50,
    show_numbers: bool = False,
    number_size: Optional[float] = None,
    number_functions: Optional[List[Optional[Callable[int, int]]]] = None,
    labels: Optional[List[str]] = None,
    label_size: Optional[float] = None,
    show_line_position: bool = False,
    spacing: float = 1
) -> None: ...

def plot_alignment_similarity_based(
    axes: Axes,
    alignment: Alignment,
    symbols_per_line: int = 50,
    show_numbers: bool = False,
    number_size: Optional[float] = None,
    number_functions: Optional[List[Optional[Callable[int, int]]]] = None,
    labels: Optional[List[str]] = None,
    label_size: Optional[float] = None,
    show_line_position: bool = False,
    spacing: float = 1,
    color: Optional[Color] = None,
    cmap: Optional[Colormap] = None,
    matrix: Optional[SubstitutionMatrix] = None,
    color_symbols: bool = False,
    symbol_size: Optional[float] = None,
    symbol_param: Optional[Dict[str, object]] = None
) -> None: ...

def plot_alignment_type_based(
    axes: Axes,
    alignment: Alignment,
    symbols_per_line: int = 50,
    show_numbers: bool = False,
    number_size: Optional[float] = None,
    number_functions: Optional[List[Optional[Callable[int, int]]]] = None,
    labels: Optional[List[str]] = None,
    label_size: Optional[float] = None,
    show_line_position: bool = False,
    spacing: float = 1,
    color_scheme: Optional[List[Color]] = None,
    color_symbols: bool = False,
    symbol_size: Optional[float] = None,
    symbol_param: Optional[Dict[str, object]] = None
) -> None: ...