# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["colors", "set_font_size_in_coord", "AdaptiveFancyArrow"]

import abc
from collections import OrderedDict
import numpy as np
from numpy.linalg import norm


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

try:
    # Only create this class when matplotlib is installed
    from matplotlib.patches import FancyArrow
    from matplotlib.patheffects import AbstractPathEffect

    class AdaptiveFancyArrow(FancyArrow):
        """
        A `FancyArrow` with fixed head shape.
        The length of the head is proportional to the width the head
        in display coordinates.
        If the head length is longer than the length of the entire
        arrow, the head length is limited to the arrow length.
        """
        
        def __init__(self, x, y, dx, dy,
                     tail_width, head_width, head_ratio, draw_head=True,
                     shape="full", **kwargs):
            if not draw_head:
                head_width = tail_width
            super().__init__(
                x, y, dx, dy,
                width=tail_width, head_width=head_width,
                overhang=0, shape=shape,
                length_includes_head=True, **kwargs
            )
            self.set_path_effects(
                [_ArrowHeadCorrect(self, head_ratio, draw_head)]
            )
        
        # Override to replace docstring
        # Removes warning:
        # unknown document: /tutorials/intermediate/constrainedlayout_guide
        def get_in_layout(self):
            """
            """
            return super().get_in_layout()
        def set_in_layout(self, in_layout):
            """
            """
            return super().set_in_layout(in_layout)
    

    class _ArrowHeadCorrect(AbstractPathEffect):
        """
        Updates the arrow head length every time the arrow is rendered
        """

        def __init__(self, arrow, head_ratio, draw_head):
            self._arrow = arrow
            self._head_ratio = head_ratio
            self._draw_head = draw_head

        def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
            # Indices to certain vertices in the arrow
            TIP = 0
            HEAD_OUTER_1 = 1
            HEAD_INNER_1 = 2
            TAIL_1 = 3
            TAIL_2 = 4
            HEAD_INNER_2 = 5
            HEAD_OUTER_2 = 6

            transform = self._arrow.axes.transData

            vert = tpath.vertices
            # Transform data coordiantes to display coordinates
            vert = transform.transform(vert)
            # The direction vector alnog the arrow
            arrow_vec = vert[TIP] - (vert[TAIL_1] + vert[TAIL_2]) / 2
            tail_width = norm(vert[TAIL_2] - vert[TAIL_1])
            # Calculate head length from head width
            head_width = norm(vert[HEAD_OUTER_2] - vert[HEAD_OUTER_1])
            head_length = head_width * self._head_ratio
            if head_length > norm(arrow_vec):
                # If the head would be longer than the entire arrow,
                # only draw the arrow head with reduced length
                head_length = norm(arrow_vec)
            # The new head start vector; is on the arrow vector
            if self._draw_head:
                head_start = \
                vert[TIP] - head_length * arrow_vec/norm(arrow_vec)
            else:
                head_start = vert[TIP]
            # vector that is orthogonal to the arrow vector
            arrow_vec_ortho = vert[TAIL_2] - vert[TAIL_1]
            # Make unit vector
            arrow_vec_ortho = arrow_vec_ortho / norm(arrow_vec_ortho)
            # Adjust vertices of the arrow head
            vert[HEAD_OUTER_1] = head_start - arrow_vec_ortho * head_width/2
            vert[HEAD_OUTER_2] = head_start + arrow_vec_ortho * head_width/2
            vert[HEAD_INNER_1] = head_start - arrow_vec_ortho * tail_width/2
            vert[HEAD_INNER_2] = head_start + arrow_vec_ortho * tail_width/2
            # Transform back to data coordinates
            # and modify path with manipulated vertices
            tpath.vertices = transform.inverted().transform(vert)
            renderer.draw_path(gc, tpath, affine, rgbFace)


except ImportError:
    pass
    

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