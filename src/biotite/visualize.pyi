# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import NewType, Optional
try:
    from matplotlib.patches import FancyArrow
    from matplotlib.text import Text
except ImportError:
    FancyArrow = NewType("FancyArrow", object)
    Text = NewType("Text", object)


def set_font_size_in_coord(
    text: Text,
    width: Optional[float] = None,
    height: Optional[float] = None,
    mode: str = "unlocked"
) -> None:


class AdaptiveFancyArrow(FancyArrow):
    def __init__(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        tail_width: float,
        head_width: float,
        head_ratio: float,
        draw_head: bool = True,
        shape: str = "full",
        **kwargs
    ) -> None: ...

colors: Dict[str, str]