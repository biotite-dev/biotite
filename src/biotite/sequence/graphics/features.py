# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["FeatureVisualizer"]

import numpy as np
from ...visualize import Visualizer
from ..annotation import Annotation, Feature, Location

class FeatureVisualizer(Visualizer):
    
    def __init__(self, annotation, loc_range=None,
                 width=800, feature_size=50, line_size=2,
                 multi_line=True, line_length=1000,
                 padding=30, border_size=10,
                 font=None, font_size=16):
        self._annotation = annotation
        self._loc_range = loc_range
        self._width = width
        self._feature_size = feature_size
        self._line_size = line_size
        self._padding = padding
        self._border_size = border_size
        self._font = font
        self._font_size = font_size
        if loc_range is None:
            self._loc_range = annotation.get_location_range()
        else:
            self._loc_range = loc_range
        if multi_line:
            self._line_length = line_length
        else:
            self._line_length = self._loc_range[1] - self._loc_range[0]
        
        self.drawfunc = {
            "CDS" : draw_cds,
            "gene" : draw_gene,
            "misc_structure" : draw_misc,
        }

    def generate(self):
        from matplotlib.patches import Rectangle

        annotation_length = self._loc_range[1] - self._loc_range[0]
        line_width = self._width - 2*self._border_size
        line_count = annotation_length // self._line_length
        # Only extend line count by 1 if there is a remainder
        if annotation_length % self._line_length != 0:
            line_count += 1
        fig_size_y = line_count * self._feature_size
        fig_size_y += (line_count-1) * self._padding
        fig_size_y += 2 * self._border_size

        fig = self.create_figure(size=(self._width, fig_size_y))

        ### Draw lines ###
        remaining_length = annotation_length
        y = fig_size_y - self._border_size - self._feature_size/2
        while remaining_length > 0:
            if remaining_length > self._line_length:
                part_line_width = line_width
            else:
                part_line_width = \
                    line_width * remaining_length / self._line_length
            line = Rectangle((self._border_size, y - self._line_size/2),
                             part_line_width, self._line_size,
                             color="gray", linewidth=0)
            fig.patches.append(line)
            y -= self._padding
            y -= self._feature_size
            remaining_length -= self._line_length
        
        ### Draw features ###
        line_start_loc = self._loc_range[0]
        y = fig_size_y - self._border_size - self._feature_size
        while line_start_loc < self._loc_range[1]:
            annotation_for_line = self._annotation[
                line_start_loc : line_start_loc + self._line_length
            ]
            for feature in annotation_for_line:
                key = feature.key
                if key == "regulatory":
                    key = feature.qual["regulatory_class"]
                if key in self.drawfunc:
                    for i in range(len(feature.locs)):
                        loc = feature.locs[i]
                        loc_len = loc.last - loc.first + 1
                        loc_in_line = loc.first - line_start_loc
                        x = self._border_size
                        x += line_width * (loc_in_line / self._line_length)
                        # Line width multiplied by percentage of line
                        width = line_width * (loc_len / self._line_length)
                        height = self._feature_size
                        self.drawfunc[key](
                            feature, x, y, width, height, fig, loc_index=i
                        )
            y -= self._padding
            y -= self._feature_size
            line_start_loc += self._line_length
        
        return fig


def draw_cds(feature, x, y, width, height, figure, loc_index=0, style_dict={}):
    def label_func(feature):
        if "product" not in feature.qual:
            return None
        elif feature.qual["product"] == "hypothetical protein":
            return None
        else:
            return feature.qual["product"]
    _draw_coding(feature, label_func, x, y, width, height, figure,
                 loc_index, style_dict)


def draw_gene(feature, x, y, width, height, figure, loc_index=0, style_dict={}):
    def label_func(feature):
        if  "gene" not in feature.qual:
            return None
        else:
            return feature.qual["gene"]
    _draw_coding(feature, label_func, x, y, width, height, figure,
                 loc_index, style_dict)


def _draw_coding(feature, label_func, x, y, width, height, figure,
                loc_index=0, style_dict={}):
    from matplotlib.patches import FancyArrow
    from matplotlib.text import Text

    head_height = 0.8*height
    tail_height = 0.5*height
    head_width = 0.4*height

    loc = feature.locs[loc_index]
    tail_x = x
    arrow_y = y + height/2
    dx = width
    dy = 0
    if head_width > width:
        head_width = width
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
                       color="forestgreen", linewidth=0)
    figure.patches.append(arrow)

    label = label_func(feature)
    if label is not None:
        center_x = x + width/2
        center_y = y + height/2
        text = Text(center_x, center_y, label,
                    color="black", ha="center", va="center",
                    size=11, figure=figure)
        figure.texts.append(text)


def draw_misc(feature, x, y, width, height, figure, loc_index=0, style_dict={}):
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text

    fraction = 0.4

    rect = Rectangle((x, y + height/2 * (1-fraction)), width, height*fraction,
                     color="black", linewidth=0)
    figure.patches.append(rect)