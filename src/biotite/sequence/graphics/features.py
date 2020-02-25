# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.graphics"
__author__ = "Patrick Kunzmann"
__all__ = ["plot_feature_map", "FeaturePlotter", "MiscFeaturePlotter",
           "CodingPlotter", "PromoterPlotter", "TerminatorPlotter",
           "RBSPlotter"]

import copy
import abc
import numpy as np
from ...visualize import colors, AdaptiveFancyArrow
from ..annotation import Annotation, Feature, Location


def plot_feature_map(axes, annotation, loc_range=None,
                     multi_line=True, symbols_per_line=1000,
                     show_numbers=False, number_size=None, line_width=0.05,
                     show_line_position=False, spacing=0.25,
                     feature_plotters=None, style_param=None):
    """
    Plot a sequence annotation, by showing the range of each feature
    on one or multiple position depicting line(s).

    This function uses :class:`FeaturePlotter` objects to draw the
    features.
    This function internally uses a list of plotters, where the
    first plotter in the list, that supports a feature, is used to draw
    that feature.
    The amount of features that can be visualized by default is limited.
    Features, that are not supported otherwise, are visualized as simple
    rectangles.
    Additional :class:`FeaturePlotter` objects, that are supplied in the
    `feature_plotters` parameters, can be used to add support for
    further features or to customize the appearance of certain features.

    Parameters
    ----------
    axes : Axes
        A *Matplotlib* axes, that is used as plotting area.
    annotation : Annotation
        The annotation to be visualized.
    loc_range : tuple (int, int), optional
        The start and exclusive stop location that is visualized.
        By default, the location range starts from the first
        base/residue and ends at the last base/residue of all features,
        ensuring that the entire annotation is drawn.
    multi_line : bool, optional
        If true, the annotation is segmented into multiple lines with a
        line break all `symbols_per_line` lines.
        Otherwise, the entire location range is put into a single line.
    symbols_per_line : int, optional
        The amount of
        Does not have an effect, if `multi_line` is false.
    show_numbers : bool, optional
        If true, the sequence position the base/residue of a line is
        shown on the right side of the plot.
    number_size : float, optional
        The font size of the position numbers
    line_width : float, optional
        The size of the continuous line as fraction of the height of
        the drawn features.
    show_line_position : bool, optional
        If true the position within a line is plotted.
    spacing : float, optional
        The size of the spacing between the lines as fraction of the
        height of the drawn features.
    feature_plotters : list of FeaturePlotter, optional
        Custom plotters for features.
        The list is iterated from the beginning until a
        :class:`FeaturePlotter` matches the respective feature
        (`FeaturePlotter.matches()` returns `True`).
        This :class:`FeaturePlotter` is then used to draw the feature.
        Therefore, the :class:`FeaturePlotter` instances in the list
        have descending priority.
        The default plotters are appended after this supplied list,
        i.e. the default plotters have a lower priority.
    style_param : dict
        Additional style parameters that are given to the
        :class:`FeaturePlotter` objects.

    Notes
    -----
    Good visulation results are obtained only for non-overlapping
    features.
    When two features overlap, their drawing area does also overlap.
    """
    from matplotlib.transforms import Bbox
    from matplotlib.patches import Rectangle

    if loc_range is None:
        loc_range = annotation.get_location_range()
    loc_range_length = loc_range[1] - loc_range[0]
    if multi_line:
        symbols_per_line = symbols_per_line
    else:
        # Line length covers the entire location range
        symbols_per_line = loc_range_length
    
    plotters = [
        PromoterPlotter(),
        TerminatorPlotter(),
        RBSPlotter(),
        CodingPlotter(),
        MiscFeaturePlotter()
    ]
    if feature_plotters is not None:
        plotters = list(feature_plotters) + plotters

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
        # Increment by spacing and width (=1) of feature
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
            plotter = None
            # Identify fitting plotter
            for potentail_plotter in plotters:
                if potentail_plotter.matches(feature):
                    # Take first fitting plotter in list
                    plotter = potentail_plotter
                    break
            if plotter is not None:
                for loc in feature.locs:
                    loc_len = loc.last - loc.first + 1
                    # Get start location realtive to start if line
                    loc_in_line = loc.first - line_start_loc
                    x = loc_in_line
                    # Line width multiplied by percentage of line
                    width = loc_len
                    height = 1
                    bbox = Bbox.from_bounds(x, y, width, height)
                    plotter.draw(
                        axes, feature, bbox, loc,
                        style_param=style_param
                    )
        # Increment by spacing and width (=1) of feature
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


class FeaturePlotter(metaclass=abc.ABCMeta):
    """
    A :class:`FeaturePlotter` is an object, that 'knows' how to draw a
    certain type of sequence feature onto a :class:`matplotlib.Axes`.

    Whether the :class:`FeaturePlotter` is able to draw a feature, is
    checked via the :func:`matches()` method.
    The visualization of a compatible feature is conducted in the
    :func:`draw()` method.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def matches(self, feature):
        """
        Check, whether this object is able to draw a given sequence
        feature.

        Parameters
        ----------
        feature : Feature
            The sequence feature to be checked.
        
        Returns
        -------
        compatibility : bool
            True, if this object is able to draw the given `feature`,
            false otherwise.
        """
        pass
    
    @abc.abstractmethod
    def draw(self, axes, feature, bbox, location, style_param):
        """
        Draw aa feature onto an axes.

        Parameters
        ----------
        axes : Axes
            A *Matplotlib* axes, that is used as plotting area.
        feature : Feature
            The feature to be drawn.
        bbox : Bbox
            The bounding box, that describes the area on the `axes`,
            where the feature should be drawn.
        location : int
            The location of the feature, that should be drawn.
            Might be useful, when the visualization is dependent
            on e.g. location defects.
        style_param : dict
            Additional style parameters.
        """
        pass


class CodingPlotter(FeaturePlotter):
    """
    A plotter for the *CDS* and *gene* features.

    Draws an arrow with a 90 degrees tip.

    Parameters
    ----------
    tail_width : float, optional
        The width of the arrow tail
        as fraction of the feature drawing area height.
    head_width : float, optional
        The width of the arrow head
        as fraction of the feature drawing area height.
    """
    
    def __init__(self, tail_width=0.5, head_width=0.8):
        self._tail_width = tail_width
        self._head_width = head_width

    def matches(self, feature):
        if feature.key in ["CDS", "gene"]:
            return True
        else:
            return False
        
    def draw(self, axes, feature, bbox, loc, style_param):
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
            x, y, dx, dy, self._tail_width, self._head_width, head_ratio=0.5,
            draw_head=draw_head, color=colors["dimgreen"], linewidth=0
        ))

        if feature.key == "CDS":
            if "product" not in feature.qual:
                label = None
            elif feature.qual["product"] == "hypothetical protein":
                label = None
            else:
                label = feature.qual["product"]
        elif feature.key == "gene":
            if  "gene" not in feature.qual:
                label = None
            else:
                label = feature.qual["gene"]
        
        if label is not None:
            center_x = bbox.x0 + bbox.width/2
            center_y = bbox.y0 + bbox.height/2
            axes.text(
                center_x, center_y, label, color="black",
                ha="center", va="center", size=11)


class MiscFeaturePlotter(FeaturePlotter):
    """
    A plotter that matches any feature.

    Draws a simple Rectangle.

    Parameters
    ----------
    height : float, optional
        The width of the rectangle
        as fraction of the feature drawing area height.
    """

    def __init__(self, height=0.4):
        self._height = height

    def matches(self, feature):
        return True
        
    def draw(self, axes, feature, bbox, loc, style_param):
        from matplotlib.patches import Rectangle

        rect = Rectangle(
            (bbox.x0, bbox.y0 + bbox.height/2 * (1-self._height)),
            bbox.width, bbox.height*self._height,
            color=colors["dimorange"], linewidth=0
        )
        axes.add_patch(rect)

class PromoterPlotter(FeaturePlotter):
    """
    A plotter for *regulatory* features with the *promoter* or
    *TATA_box* class.

    Draws a simple curved thin black arrow.

    Parameters
    ----------
    line_width : float, optional
        The width of the curved arrow tail.
    head_width : float, optional
        The width of the arrow head
    head_length : float, optional
        The length of the arrow.
    head_height : float, optional
        The Y-position of the arrow head
        as fraction of the halffeature drawing area height.
    """

    def __init__(self, line_width=2, head_width=2,
                 head_length=6, head_height=0.8):
        self._line_width = line_width
        self._head_width = head_width
        self._head_length = head_length
        self._head_height = head_height

    def matches(self, feature):
        if feature.key == "regulatory":
            if "regulatory_class" in feature.qual:
                if feature.qual["regulatory_class"] in ["promoter","TATA_box"]:
                    return True
        return False
        
    def draw(self, axes, feature, bbox, loc, style_param):
        from matplotlib.patches import FancyArrowPatch, ArrowStyle
        from matplotlib.path import Path

        x_center = bbox.x0 + bbox.width/2
        y_center = bbox.y0 + bbox.height/2

        path = Path(
            vertices=[
                (bbox.x0, y_center),
                (bbox.x0, y_center - bbox.height/2 * self._head_height),
                (bbox.x1, y_center - bbox.height/2 * self._head_height),
            ],
            codes=[
                Path.MOVETO,
                Path.CURVE3,
                Path.CURVE3
            ]
        )
        style = ArrowStyle.CurveFilledB(
            head_width=self._head_width, head_length=self._head_length
        )
        arrow = FancyArrowPatch(
            path=path, arrowstyle=style, linewidth=self._line_width,
            color="black"
        )
        axes.add_patch(arrow)
        
        if "note" in feature.qual:
            axes.text(
                x_center, y_center + bbox.height/4, feature.qual["note"],
                color="black", ha="center", va="center",
                size=9
            )


class TerminatorPlotter(FeaturePlotter):
    """
    A plotter for *regulatory* features with the *terminator* class.

    Draws a vertical bar.

    Parameters
    ----------
    bar_width : float, optional
        The width of the line representing the bar.
    """

    def __init__(self, bar_width=5):
        self._bar_width = bar_width

    def matches(self, feature):
        if feature.key == "regulatory":
            if "regulatory_class" in feature.qual:
                if feature.qual["regulatory_class"] == "terminator":
                    return True
        return False
        
    def draw(self, axes, feature, bbox, loc, style_param):

        x = bbox.x0 + bbox.width/2

        axes.plot(
            (x, x), (bbox.y0, bbox.y1), color="black",
            linestyle="-", linewidth=self._bar_width, marker="None"
        )


class RBSPlotter(FeaturePlotter):
    """
    A plotter for *regulatory* features with the
    *ribosome_binding_site* class.

    Draws an ellipse.

    Parameters
    ----------
    height : float, optional
        The width of the ellipse
        as fraction of the feature drawing area height.
    """

    def __init__(self, height=0.4):
        self._height = height

    def matches(self, feature):
        if feature.key == "regulatory":
            if "regulatory_class" in feature.qual:
                if feature.qual["regulatory_class"] == "ribosome_binding_site":
                    return True
        return False
        
    def draw(self, axes, feature, bbox, loc, style_param):
        from matplotlib.patches import Ellipse

        ellipse = Ellipse(
            (bbox.x0 + bbox.width/2, bbox.y0 + bbox.height/2),
            bbox.width, self._height*bbox.height,
            color=colors["dimorange"], linewidth=0)
        axes.add_patch(ellipse)