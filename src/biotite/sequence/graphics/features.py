# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["plot_feature_map"]

import copy
import numpy as np
from ...visualize import colors, AdaptiveFancyArrow
from ..annotation import Annotation, Feature, Location


def plot_feature_map(axes, annotation, loc_range=None,
                     multi_line=True, symbols_per_line=1000,
                     show_numbers=False, number_size=None, line_width=0.05,
                     show_line_position=False, spacing=0.25, draw_functions=None,
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
    head_width = 0.8*bbox.height
    tail_width = 0.5*bbox.height

    loc = feature.locs[loc_index]
    y = bbox.y0 + bbox.height/2
    dy = 0
    if loc.strand == Location.Strand.FORWARD:
        x = bbox.x0
        dx = bbox.width
    else:
        x = bbox.x1
        dx = -bbox.width
    
    if  (
            loc.strand == Location.Strand.FORWARD 
            and loc.defect & Location.Defect.MISS_RIGHT
        ) or (
            loc.strand == Location.Strand.REVERSE 
            and loc.defect & Location.Defect.MISS_LEFT
        ):
            # If the feature extends into the prevoius or next line
            # do not draw an arrow head
            draw_head = False
    else:
            draw_head = True
    
    # Create head with 90 degrees tip -> head width/length ratio = 1/2
    axes.add_patch(AdaptiveFancyArrow(
        x, y, dx, dy, tail_width, head_width, head_ratio=0.5,
        draw_head=draw_head, color=colors["dimgreen"], linewidth=0
    ))

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
    from matplotlib.patches import FancyArrowPatch, ArrowStyle
    from matplotlib.path import Path

    line_width = 2
    head_width = 2
    head_length = 6
    head_height = 0.8

    x_center = bbox.x0 + bbox.width/2
    y_center = bbox.y0 + bbox.height/2

    path = Path(
        vertices=[
            (bbox.x0, y_center),
            (bbox.x0, y_center - bbox.height/2 * head_height),
            (bbox.x1, y_center - bbox.height/2 * head_height),
        ],
        codes=[
            Path.MOVETO,
            Path.CURVE3,
            Path.CURVE3
        ]
    )
    style = ArrowStyle.CurveFilledB(
        head_width=head_width, head_length=head_length
    )
    arrow = FancyArrowPatch(
        path=path, arrowstyle=style, linewidth=line_width,
        color="black"
    )
    axes.add_patch(arrow)
    
    if "note" in feature.qual:
        axes.text(
            x_center, y_center + bbox.height/4, feature.qual["note"],
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