# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

from ..features import *
from ..annotation import Location
import matplotlib.patches as patches

__all__ = ["draw_cds"]


def draw_cds(axes, feature, x, y, width, height, defect, dir, **kwargs):
    # Thickness of arrow base
    b_height = height*0.6
    # Thickness pf arrow head
    h_height = height*0.9
    if width >= h_height/2:
        h_width = h_height/2
    else:
        h_width = width
    if dir == "right":
        start_x = x
        dx = width
    else:
        start_x = x + width
        dx = -width
    axes.add_patch(patches.FancyArrow(
        start_x, y, dx, 0,
        width=b_height, head_width=h_height, head_length=h_width,
        length_includes_head=True, facecolor="green", edgecolor="None")
    )
    axes.text(x+width/2, y, feature.get_gene(), fontsize=3,
              ha="center", va="center")