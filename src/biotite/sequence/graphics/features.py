# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["FeatureMap"]

import numpy as np
from ...visualize import Visualizer, colors
from ..annotation import Annotation, Feature, Location

class FeatureMap(Visualizer):
    """
    A `FeatureMap` is a location based visualization of sequence
    features.

    EXPERIMENTAL: Future API changes are probable.

    Parameters
    ----------
    annotation : Annotation
        This annotation contains the features to be visualized.
    loc_range : tuple of int, length=2, optional
        The displayed location range. By default the entire range,
        i.e. the first base/residue of the first feature and the last
        base/residue of the last feature is taken as range.
    width : float, optional
        The width of the figure (pixels). (Default: 800)
    multi_line : bool, optional
        If true, display the features in multiple lines. Otherwise
        display the features on a single line.
    line_length : int
        The amount of bases/residues per line. (Default: 1000)
    
    Attributes
    ----------
    drawfunc : dict
        A dictionary that maps sequence feature keys to functions,
        that draw a representation for that feature class on
        the figure.
        Features with keys that are not in this dictionary are not
        drawn. Add or modify the items to extend or alter the
        visualization.
    """

    def __init__(self, annotation, loc_range=None,
                 width=800, multi_line=True, line_length=1000):
        super().__init__()
        self._annotation   = annotation
        self._loc_range    = loc_range
        self._width        = width

        self._feature_size = 50
        self._line_size    = 2
        
        self._spacing      = 30
        
        self._margin  = 10
        
        self._show_numbers        = False
        self._number_size         = 150
        self._number_font         = None
        self._number_font_size    = 16
        
        if loc_range is None:
            self._loc_range = annotation.get_location_range()
        else:
            self._loc_range = loc_range
        if multi_line:
            self._line_length = line_length
        else:
            self._line_length = self._loc_range[1] - self._loc_range[0]
        
        self.drawfunc = {
            "CDS"                   : draw_cds,
            "gene"                  : draw_gene,
            "misc_structure"        : draw_misc,
            "promoter"              : draw_promoter,
            "terminator"            : draw_terminator,
            "ribosome_binding_site" : draw_rbs
        }

        self.style = {}
    
    def add_location_numbers(self, size=150, font_size=16, font=None):
        """
        Add numbers to the right side of the figure, that display the
        respective sequence location at the right end of the line.

        Parameters
        ----------
        size : float, optional
            The size of the number column in x-direction (pixels).
            (Default: 50)
        font_size : float, optional
            Font size of the numbers.
            (Default: 16)
        font : FontProperties, optional
            `matplotlib` `FontProperties` for customization of the
            font used by the numbers.
        """
        self._show_numbers     = True
        self._number_size      = size
        self._number_font      = font
        self._number_font_size = font_size
    
    def set_size(self, feature_size=50, line_size=2):
        """
        Set the size of the features and the guiding line.

        Parameters
        ----------
        feature_size : float, optional
            The height of a feature's drwaing area (pixels).
            (Default: 50)
        line_size : float, optional
            The height of the guidling line (pixels).
            (Default: 2)
        """
        self._feature_size = feature_size
        self._line_size    = line_size

    def set_spacing(self, spacing):
        """
        Set the spacing between the lines.

        Parameters
        ----------
        spacing : float
            The spacing between the lines.
        """
        self._spacing = spacing
    
    def set_margin(self, margin):
        """
        Set the margin of the figure.

        Parameters
        ----------
        margin : float
            The margin of the figure.
        """
        self._margin = margin

    def generate(self):
        from matplotlib.patches import Rectangle
        from matplotlib.text import Text

        annotation_length = self._loc_range[1] - self._loc_range[0]
        line_width = self._width - 2*self._margin
        if self._show_numbers:
            line_width -= self._number_size
        line_count = annotation_length // self._line_length
        # Only extend line count by 1 if there is a remainder
        if annotation_length % self._line_length != 0:
            line_count += 1
        fig_size_y = line_count * self._feature_size
        fig_size_y += (line_count-1) * self._spacing
        fig_size_y += 2 * self._margin

        fig = self.create_figure(size=(self._width, fig_size_y))

        ### Draw lines ###
        remaining_length = annotation_length
        y = fig_size_y - self._margin - self._feature_size/2
        while remaining_length > 0:
            if remaining_length > self._line_length:
                part_line_width = line_width
            else:
                part_line_width = \
                    line_width * remaining_length / self._line_length
            line = Rectangle((self._margin, y - self._line_size/2),
                             part_line_width, self._line_size,
                             color="gray", linewidth=0)
            fig.patches.append(line)
            y -= self._spacing
            y -= self._feature_size
            remaining_length -= self._line_length
        
        ### Draw features ###
        line_start_loc = self._loc_range[0]
        y = fig_size_y - self._margin - self._feature_size
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
                        x = self._margin
                        x += line_width * (loc_in_line / self._line_length)
                        # Line width multiplied by percentage of line
                        width = line_width * (loc_len / self._line_length)
                        height = self._feature_size
                        self.drawfunc[key](
                            feature, x, y, width, height, fig, loc_index=i,
                            style_dict=self.style
                        )
            y -= self._spacing
            y -= self._feature_size
            line_start_loc += self._line_length
        
        ### Draw position numbers ###
        if self. _show_numbers:
            x = self._width - self._margin
            y = fig_size_y - self._margin - self._feature_size/2
            for i in range(line_count):
                if i == line_count-1:
                    # Last line -> get number of last column in trace
                    loc = self._loc_range[1] -1
                else:
                    loc = self._loc_range[0] + ((i+1) * self._line_length) -1
                text = Text(x, y, str(loc),
                            color="black", ha="right", va="center",
                            size=self._number_font_size, figure=fig,
                            fontproperties=self._number_font)
                fig.texts.append(text)
                y -= self._feature_size
                y -= self._spacing


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
                            x = self._margin
                            x += line_width * (loc_in_line / self._line_length)
                            # Line width multiplied by percentage of line
                            width = line_width * (loc_len / self._line_length)
                            height = self._feature_size
                            self.drawfunc[key](
                                feature, x, y, width, height, fig, loc_index=i
                            )
                y -= self._spacing
                y -= self._feature_size
                line_start_loc += self._line_length
        
        return fig




def draw_cds(feature, x, y, width, height, figure, loc_index, style_dict):
    def label_func(feature):
        if "product" not in feature.qual:
            return None
        elif feature.qual["product"] == "hypothetical protein":
            return None
        else:
            return feature.qual["product"]
    _draw_coding(feature, label_func, x, y, width, height, figure,
                 loc_index, style_dict)


def draw_gene(feature, x, y, width, height, figure, loc_index, style_dict):
    def label_func(feature):
        if  "gene" not in feature.qual:
            return None
        else:
            return feature.qual["gene"]
    _draw_coding(feature, label_func, x, y, width, height, figure,
                 loc_index, style_dict)


def _draw_coding(feature, label_func, x, y, width, height, figure,
                loc_index, style_dict):
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
                       color=colors["dimgreen"], linewidth=0)
    figure.patches.append(arrow)

    label = label_func(feature)
    if label is not None:
        center_x = x + width/2
        center_y = y + height/2
        text = Text(center_x, center_y, label,
                    color="black", ha="center", va="center",
                    size=11, figure=figure)
        figure.texts.append(text)


def draw_misc(feature, x, y, width, height, figure, loc_index, style_dict):
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text

    fraction = 0.4

    rect = Rectangle((x, y + height/2 * (1-fraction)), width, height*fraction,
                     color="black", linewidth=0)
    figure.patches.append(rect)


def draw_promoter(feature, x, y, width, height, figure, loc_index, style_dict):
    from matplotlib.patches import Rectangle
    from matplotlib.patches import Wedge
    from matplotlib.patches import FancyArrow
    from matplotlib.text import Text

    base_width = 4
    head_width = 8
    curve_radius = 10

    y_center = y + height/2
    y_curve_max = y + height - head_width/2 + base_width/2
    y_curve_min = y_curve_max - curve_radius
    y_tip = y + height - head_width/2

    vertical = Rectangle((x, y_center), base_width, y_curve_min-y_center,
                         color="black", linewidth=0)
    figure.patches.append(vertical)
    curve = Wedge(center=(x+curve_radius, y_curve_min), r=curve_radius,
                  theta1=90, theta2=180, width=base_width,
                  color="black", linewidth=0)
    figure.patches.append(curve)
    horizontal = FancyArrow(
        x+curve_radius, y_tip, dx=width-curve_radius, dy=0,
        width=base_width, head_width=head_width, head_length=head_width,
        length_includes_head=True, color="black", linewidth=0
    )
    figure.patches.append(horizontal)
    if "note" in feature.qual:
        text = Text(x + width/2, y + height/4, feature.qual["note"],
                    color="black", ha="center", va="center",
                    size=9, figure=figure)
        figure.texts.append(text)


def draw_terminator(feature, x, y, width, height, figure,
                    loc_index, style_dict):
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text

    bar_width = 5
    x_start = x + width/2 - bar_width/2

    rect = Rectangle((x_start, y), bar_width, height,
                     color="black", linewidth=0)
    figure.patches.append(rect)

def draw_rbs(feature, x, y, width, height, figure, loc_index, style_dict):
    from matplotlib.patches import Ellipse

    fraction = 0.4

    ellipse = Ellipse((x + width/2, y + height/2), width, fraction*height,
                     color=colors["dimorange"], linewidth=0)
    figure.patches.append(ellipse)