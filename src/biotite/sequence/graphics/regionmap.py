# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

from .features import *
from ..annotation import Location
import copy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

__all__ = ["draw_region_map", "get_default_map_style"]


default_style = {
    "strandsize"        : 0.03,
    "strandcolor"       : "black",
    "fontsize"          : 0.15,
    "cds"               : {
        "arrowcolor"        : "limegreen",
        "fontcolor"         : "black"
    },
    "drawers"           : {
        "CDS"        : draw_cds,
        "promoter"   : draw_promoter,
        "terminator" : draw_terminator,
    },
}

def get_default_map_style():
    return copy.deepcopy(default_style)

def draw_region_map(annotation, region, size, margin,
                    dpi=96, spacing=0, bases_per_line=1000,
                    style=default_style):
    width = size[0]
    height = size[1]
    fig = plt.figure(figsize=(width/float(dpi), height/float(dpi)), dpi=dpi)
    line_width = width - 2*margin[0]
    line_height = height - 2*margin[1] - spacing
    line_x = margin[0]
    line_y = margin[1] + spacing/2
    
    # Draw strand marker
    strand_height = line_height * style["strandsize"]
    strand_marker_x = line_x
    strand_marker_y = line_y - strand_height/2 + line_height/2
    fig.patches.append(Rectangle(
        (strand_marker_x, strand_marker_y), line_width, strand_height,
        edgecolor="None", facecolor=style["strandcolor"], figure=fig
    ))
    
    # Draw Annotations
    annotation = annotation[region[0]:region[1]]
    region_first = region[0]
    region_last = region[1]
    region_span = region_last - region_first +1
    width_per_base = line_width / region_span
    for feature in annotation:
        for loc in feature.locs:
            # Determine drawing area for feature location
            feature_span = loc.last - loc.first +1
            feature_x = line_x + (loc.first-region_first) * width_per_base
            feature_y = line_y
            feature_width = width_per_base * feature_span
            feature_height = line_height
            if feature_x < line_x:
                # Prevent feature from drawing out of bounds (left)
                diff = line_x - feature_x
                feature_x += diff
                feature_width -= diff
            if feature_x + feature_width > line_x + line_width:
                # Prevent feature from drawing out of bounds (right)
                feature_width = line_width + line_x - feature_x
            # Determine the drawing direction
            if loc.strand == Location.Strand.FORWARD:
                reverse = False
            else:
                reverse = True
            # Call feature specific draw function
            feature_key = feature.key
            if feature_key == "regulatory":
                feature_key = feature.qual["regulatory_class"]
            try:
                draw_func = style["drawers"][feature_key]
            except KeyError:
                continue
            draw_func(fig, feature,
                      feature_x, feature_y+feature_height/2,
                      feature_width, feature_height,
                      style["fontsize"]*feature_height, reverse, style)
    
    return fig


