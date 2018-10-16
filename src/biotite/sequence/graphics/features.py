# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["plot_feature_map"]

import copy
import numpy as np
from ...visualize import colors
from ..annotation import Annotation, Feature, Location


def plot_feature_map(axes, annotation, loc_range=None,
                     multi_line=True, symbols_per_line=1000,
                     show_numbers=False, number_size=None, line_width=0.05,
                     show_line_position=False, spacing=0.5, draw_functions=None,
                     style_param=None):
    from matplotlib.transforms import Bbox
    from matplotlib.patches import Rectangle
    
    if loc_range is None:
        loc_range = annotation.get_location_range()
    else:
        loc_range = loc_range
    loc_range_length = loc_range[1] - loc_range[0]
    if multi_line:
        symbols_per_line = symbols_per_line
    else:
        # Line length covers the entire location range
        symbols_per_line = loc_range_length
    
    drawfunc = copy.copy(DRAW_FUNCTIONS)
    if draw_functions is not None:
        for annot_type, func in draw_functions.items():
            drawfunc[annot_type] = func

    style_param = {} if style_param is None else style_param

    line_count = loc_range_length // symbols_per_line
    # Only extend line count by 1 if there is a remainder
    if loc_range_length % symbols_per_line != 0:
        line_count += 1

    
    ### Draw lines ###
    remaining_symbols = loc_range_length
    y = 0.5
    while remaining_symbols > 0:
        if remaining_symbols > symbols_per_line:
            # Line spans the entire plot (horizontally)
            line_length = symbols_per_line
        else:
            # Last line -> Line spans to end of annotation
            line_length = remaining_symbols
        axes.add_patch(Rectangle(
            (0, y-line_width/2), line_length, line_width,
            color="gray", linewidth=0
        ))
        # Increment by spacing and width of feature (1)
        y += spacing + 1
        remaining_symbols -= symbols_per_line
    
    ### Draw features ###
    line_start_loc = loc_range[0]
    y = 0
    while line_start_loc < loc_range[1]:
        annotation_for_line = annotation[
            line_start_loc : line_start_loc + symbols_per_line
        ]
        for feature in annotation_for_line:
            key = feature.key
            if key == "regulatory":
                key = feature.qual["regulatory_class"]
            if key in drawfunc and drawfunc[key] is not None:
                for i in range(len(feature.locs)):
                    loc = feature.locs[i]
                    loc_len = loc.last - loc.first + 1
                    # Get start location realtive to start if line
                    loc_in_line = loc.first - line_start_loc
                    x = loc_in_line
                    # Line width multiplied by percentage of line
                    width = loc_len
                    height = 1
                    bbox = Bbox.from_bounds(x, y, width, height)
                    drawfunc[key](
                        axes, feature, bbox, loc_index=i,
                        style_param=style_param
                    )
        # Increment by spacing and width of feature (1)
        y += spacing + 1
        remaining_symbols += symbols_per_line
        line_start_loc += symbols_per_line
    
    ### Draw position numbers  ###
    ticks = []
    tick_labels = []
    if show_numbers:
        # Numbers at center height of each feature line -> 0.5
        y = 0.5
        for i in range(line_count):
            if i == line_count-1:
                # Last line -> get number of last column in trace
                loc = loc_range[1] -1
            else:
                loc = loc_range[0] + ((i+1) * symbols_per_line) -1
            ticks.append(y)
            tick_labels.append(str(loc))
            # Increment by spacing and width of feature (1)
            y += spacing + 1
    axes.set_yticks(ticks)
    axes.set_yticklabels(tick_labels)


    axes.set_xlim(0, symbols_per_line)
    # Y-axis starts from top
    axes.set_ylim(1*line_count + spacing*(line_count-1), 0)
    axes.set_frame_on(False)
    # Draw location numbers on right side
    axes.get_yaxis().set_tick_params(
        left=False, right=False, labelleft=False, labelright=True
    )
    # Remove ticks and set number font size
    axes.yaxis.set_tick_params(
        left=False, right=False, labelsize=number_size
    )
    
    if show_line_position:
        axes.xaxis.set_tick_params(
            top=False, bottom=True, labeltop=False, labelbottom=True
        )
    else:
        axes.xaxis.set_tick_params(
            top=False, bottom=False, labeltop=False, labelbottom=False
        )


def draw_cds(axes, feature, bbox, loc_index, style_param):
    def label_func(feature):
        if "product" not in feature.qual:
            return None
        elif feature.qual["product"] == "hypothetical protein":
            return None
        else:
            return feature.qual["product"]
    _draw_coding(axes, feature, bbox, loc_index, style_param, label_func)


def draw_gene(axes, feature, bbox, loc_index, style_param):
    def label_func(feature):
        if  "gene" not in feature.qual:
            return None
        else:
            return feature.qual["gene"]
    _draw_coding(axes, feature, bbox, loc_index, style_param, label_func)


def _draw_coding(axes, feature, bbox, loc_index, style_param, label_func):
    from matplotlib.patches import FancyArrow

    head_height = 0.8*bbox.height
    tail_height = 0.5*bbox.height
    head_width = 0.4*bbox.height

    loc = feature.locs[loc_index]
    tail_x = bbox.x0
    arrow_y = bbox.y0 + bbox.height/2
    dx = bbox.width
    dy = 0
    if head_width > bbox.width:
        head_width = bbox.width
    if loc.strand == Location.Strand.REVERSE:
        tail_x += dx
        dx = -dx
    if loc.strand == Location.Strand.FORWARD \
        and loc.defect & Location.Defect.MISS_RIGHT:
            head_width = 0
            head_height = tail_height
    if loc.strand == Location.Strand.REVERSE \
        and loc.defect & Location.Defect.MISS_LEFT:
            head_width = 0
            head_height = tail_height
    arrow = FancyArrow(tail_x, arrow_y, dx, dy,
                       width=tail_height, head_width=head_height,
                       head_length=head_width, length_includes_head=True,
                       color=colors["dimgreen"], linewidth=0)
    axes.add_patch(arrow)

    label = label_func(feature)
    if label is not None:
        center_x = bbox.x0 + bbox.width/2
        center_y = bbox.y0 + bbox.height/2
        axes.text(
            center_x, center_y, label, color="black",
            ha="center", va="center", size=11)


def draw_misc(axes, feature, bbox, loc_index, style_param):
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text

    fraction = 0.4

    rect = Rectangle(
        (bbox.x0, bbox.y0 + bbox.height/2 * (1-fraction)),
        bbox.width, bbox.height*fraction,
        color="black", linewidth=0
    )
    axes.add_patch(rect)


def draw_promoter(axes, feature, bbox, loc_index, style_param):
    from matplotlib.patches import Rectangle
    from matplotlib.patches import Wedge
    from matplotlib.patches import FancyArrow
    from matplotlib.text import Text

    base_width = 0.08
    head_width = 0.15
    curve_radius = 0.20

    y_center = bbox.y0 + bbox.height/2
    y_curve_max = bbox.y1 - bbox.height*head_width/2 + bbox.height*base_width/2
    y_curve_min = y_curve_max - bbox.height*curve_radius
    y_tip = bbox.y0 + bbox.height - bbox.height*head_width/2

    vertical = Rectangle((bbox.x0, y_center), base_width, y_curve_min-y_center,
                         color="black", linewidth=0)
    axes.add_patch(vertical)
    curve = Wedge(
        center=(bbox.x0+curve_radius, y_curve_min), r=curve_radius,
        theta1=90, theta2=180, width=base_width,
        color="black", linewidth=0
    )
    axes.add_patch(curve)
    horizontal = FancyArrow(
        bbox.x0+curve_radius, y_tip, dx=bbox.width-curve_radius, dy=0,
        width=base_width, head_width=head_width, head_length=head_width,
        length_includes_head=True, color="black", linewidth=0
    )
    axes.add_patch(horizontal)
    if "note" in feature.qual:
        axes.text(
            bbox.x0 + bbox.width/2, bbox.y0 + bbox.height/4, feature.qual["note"],
            color="black", ha="center", va="center",
            size=9
        )


def draw_terminator(axes, feature, bbox, loc_index, style_param):
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text

    bar_width = 5
    x = bbox.x0 + bbox.width/2

    axes.plot(
        (x, x), (bbox.y0, bbox.y1),
        color="black", linestyle="-", linewidth=bar_width, marker="None"
    )

def draw_rbs(axes, feature, bbox, loc_index, style_param):
    from matplotlib.patches import Ellipse

    fraction = 0.4

    ellipse = Ellipse(
        (bbox.x0 + bbox.width/2, bbox.y0 + bbox.height/2),
        bbox.width, fraction*bbox.height,
        color=colors["dimorange"], linewidth=0)
    axes.add_patch(ellipse)


DRAW_FUNCTIONS = {
    "CDS"                   : draw_cds,
    "gene"                  : draw_gene,
    "misc_structure"        : draw_misc,
    "promoter"              : draw_promoter,
    "terminator"            : draw_terminator,
    "ribosome_binding_site" : draw_rbs
}