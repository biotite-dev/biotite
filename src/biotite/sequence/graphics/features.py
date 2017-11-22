# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

from ..annotation import Location
from matplotlib.patches import FancyArrow, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import IdentityTransform

__all__ = ["draw_cds", "draw_promoter", "draw_terminator"]


def draw_cds(fig, feature, x, y, width, height, fontsize, reverse, style):
    cds_style = style["cds"]
    # Thickness of arrow base
    b_height = height*0.5
    # Thickness pf arrow head
    h_height = height*0.9
    if width >= h_height/2:
        h_width = h_height/2
    else:
        h_width = width
    if reverse == True:
        start_x = x + width
        dx = -width
    else:
        start_x = x
        dx = width
    fig.patches.append(FancyArrow(
        start_x, y, dx, 0,
        width=b_height, head_width=h_height, head_length=h_width,
        length_includes_head=True,
        facecolor=cds_style["arrowcolor"], edgecolor="None",
        figure=fig
    ))
    if "gene" in feature.qual:
        fig.texts.append(Text(
            x+width/2, y, feature.qual["gene"],
            fontsize=fontsize, color=cds_style["fontcolor"],
            ha="center", va="center", figure=fig
        ))


def draw_promoter(fig, feature, x, y, width, height, fontsize, reverse, style):
    line_width = height * 0.05
    arrow_height = height * 0.4
    # Vertical segment
    dx_rect = line_width
    dy_rect = arrow_height
    fig.patches.append(Rectangle(
        (x, y), dx_rect, dy_rect,
        edgecolor="None", facecolor="black", figure=fig
    ))
    # horizontal segment
    fig.patches.append(FancyArrow(
        x, y+arrow_height, width, 0,
        width=line_width, head_width=3*line_width, head_length=line_width*3,
        overhang=0.2, length_includes_head=True,
        facecolor="black", edgecolor="None",
        figure=fig
    ))


def draw_terminator(fig, feature, x, y, width, height, fontsize, reverse,
                    style):
    dx = height * 0.1
    dy = height * 0.8
    if dx > width:
        dx = width
    if reverse:
        dy_for, dy_rev = (dy_rev, dy_for)
    fig.patches.append(Rectangle(
        (x + width/2 - dx/2, y - dy/2), dx, dy,
        edgecolor="None", facecolor="gray", figure=fig
    ))
    