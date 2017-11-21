# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

from ..features import *
from ..annotation import Location
from matplotlib.patches import FancyArrow
from matplotlib.text import Text
from matplotlib.transforms import IdentityTransform

__all__ = ["draw_cds"]


def draw_cds(fig, feature, x, y, width, height, fontsize, direction, style):
    cds_style = style["cds"]
    # Thickness of arrow base
    b_height = height*0.5
    # Thickness pf arrow head
    h_height = height*0.9
    if width >= h_height/2:
        h_width = h_height/2
    else:
        h_width = width
    if direction == "right":
        start_x = x
        dx = width
    else:
        start_x = x + width
        dx = -width
    fig.patches.append(FancyArrow(
        start_x, y, dx, 0,
        width=b_height, head_width=h_height, head_length=h_width,
        length_includes_head=True,
        facecolor=cds_style["arrowcolor"], edgecolor="None",
        figure=fig
    ))
    fig.texts.append(Text(
        x+width/2, y, feature.get_gene(),
        fontsize=fontsize, color=cds_style["fontcolor"],
        ha="center", va="center", figure=fig
    ))