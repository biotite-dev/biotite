# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from ..features import *
from ..annotation import Location
import matplotlib.patches as patches

__all__ = ["draw_cds"]


def draw_cds(axes, feature, x, y, width, height, defect, dir, **kwargs):
    base_height = height*0.6
    head_height = height*0.9
    if width >= head_height/2:
        head_width = head_height/2
        base_width = width-head_width
    else:
        head_width = width
        base_width = None
    if base_width is not None:
        if dir == "right":
            axes.add_patch(patches.FancyArrow(
                x, y-base_height/2, width, 0,
                width=base_height, head_width=head_height, head_length=head_width,
                length_includes_head=True, facecolor="green", edgecolor="None")
            )
        else:
            pass
    """
    axes.add_patch(patches.Rectangle(
        (x+head_width, y-base_height/2), base_width, base_height*0.5,
        facecolor="green")
    )
    """