# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["colors", "set_font_size_in_coord", "draw_adaptive_arrow"]

import abc
from collections import OrderedDict
import numpy as np


def set_font_size_in_coord(text, width=None, height=None, mode="unlocked"):
    from matplotlib.transforms import Bbox
    from matplotlib.text import Text
    from matplotlib.patheffects import AbstractPathEffect

    class TextScaler(AbstractPathEffect):
        def __init__(self, text, width, height, mode):
            self._text = text
            self._mode = mode
            self._width = width
            self._height = height

        def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
            ax = self._text.axes
            renderer = ax.get_figure().canvas.get_renderer()
            bbox = text.get_window_extent(renderer=renderer)
            bbox = Bbox(ax.transData.inverted().transform(bbox))
            
            if self._mode == "proportional":
                if self._width is None:
                    # Proportional scaling based on height
                    scale_y = self._height / bbox.height
                    scale_x = scale_y
                elif self._height is None:
                    # Proportional scaling based on width
                    scale_x = self._width / bbox.width
                    scale_y = scale_x
            elif self._mode == "unlocked":
                scale_x = self._width / bbox.width
                scale_y = self._height / bbox.height
            elif self._mode == "minimum":
                scale_x = self._width / bbox.width
                scale_y = self._height / bbox.height
                scale = max(scale_x, scale_y)
                scale_x, scale_y = scale, scale
            elif self._mode == "maximum":
                scale_x = self._width / bbox.width
                scale_y = self._height / bbox.height
                scale = min(scale_x, scale_y)
                scale_x, scale_y = scale, scale

            affine = affine.identity().scale(scale_x, scale_y) + affine
            renderer.draw_path(gc, tpath, affine, rgbFace)
    
    if mode in ["unlocked", "minimum", "maximum"]:
        if width is None or height is None:
            raise TypeError(
                f"Width and height must be set in '{mode}' mode"
            )
    elif mode == "proportional":
        if  not (width  is None and height is not None) or \
            not (height is None and width  is not None):
                raise TypeError(
                    f"Either width or height must be set in '{mode}' mode"
                )
    else:
        raise ValueError(
                f"Unknown mode '{mode}'"
            )
    text.set_path_effects([TextScaler(text, width, height, mode)])


def draw_adaptive_arrow(axes, x, y, dx, dy,
                        tail_width, head_width, head_ratio, draw_head=True,
                        shape="full", **kwargs):
    from matplotlib.patches import FancyArrow
    from matplotlib.transforms import Bbox

    arrow = None
    
    def on_draw(event):
        """
        Callback function that is called, every time the figure is resized
        Removes the current arrow and replaces it with an arrow with
        recalcualted head
        """
        nonlocal tail_width
        nonlocal head_width
        nonlocal arrow
        if arrow is not None:
            arrow.remove()
        # Create a head that looks equal, independent of the aspect
        # ratio
        # Hence, a transformation into display coordinates has to be
        # performed to fix the head width to length ratio
        # In this transformation only the height and width are
        # interesting, absolute coordinates are not needed
        # -> box origin at (0,0)
        arrow_box = Bbox([(0,0),(0,head_width)])
        arrow_box_display = axes.transData.transform_bbox(arrow_box)
        head_length_display = np.abs(arrow_box_display.height * head_ratio)
        arrow_box_display.x1 = arrow_box_display.x0 + head_length_display
        # Transfrom back to data coordinates for plotting
        arrow_box = axes.transData.inverted().transform_bbox(arrow_box_display)
        head_length = arrow_box.width
        if head_length > np.abs(dx):
            # If the head would be longer than the entire arrow,
            # only draw the arrow head with reduced length
            head_length = np.abs(dx)
        if not draw_head:
            head_length = 0
            head_width = tail_width
        arrow = FancyArrow(
            x, y, dx, dy,
            width=tail_width, head_width=head_width, head_length=head_length,
            length_includes_head=True, **kwargs)
        axes.add_patch(arrow)
    
    # Register callback function
    axes.get_figure().canvas.mpl_connect("resize_event", on_draw)
    


# Biotite themed colors
colors = OrderedDict([
    ("brightorange" , "#ffb569ff"),
    ("lightorange"  , "#ff982dff"),
    ("orange"       , "#ff8405ff"),
    ("dimorange"    , "#dc7000ff"),
    ("darkorange"   , "#b45c00ff"),
    ("brightgreen"  , "#98e97fff"),
    ("lightgreen"   , "#6fe04cff"),
    ("green"        , "#52da2aff"),
    ("dimgreen"     , "#45bc20ff"),
    ("darkgreen"    , "#389a1aff"),
])