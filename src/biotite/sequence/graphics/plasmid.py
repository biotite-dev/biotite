# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.graphics"
__author__ = "Patrick Kunzmann"
__all__ = ["plot_plasmid_map"]

import copy
import warnings
import abc
import numpy as np
import re
from ...visualize import colors
from ..annotation import Annotation, Feature, Location


def plot_plasmid_map(axes, annotation, plasmid_size, tick_length=0.02,
                     tick_step=200, ring_width=0.01, feature_width=0.06,
                     spacing=0.01, arrow_head_length=0.04, label=None,
                     face_properties=None, label_properties=None,
                     omit_oversized_labels=True, feature_formatter=None):
    """
    Plot a plasmid map using the sequence features in the given
    :class:`Annotation`.

    Each feature location is depicted either by a curved arrow,
    indicating on which strand the feature resides, or an arc.
    The ticks indicate the sequence position.

    **EXPERIMENTAL**: The API and the resulting plots will probably
    change in future versions.

    Parameters
    ----------
    axes : PolarAxes
        A *Matplotlib* axes, that is used as plotting area.
        A polar projection is required.
    annotation : Annotation
        The annotation to be visualized.
    plasmid_size : int
        The size of the plasmid, i.e. the length of its sequence.
    tick_length : float, optional
        The length of the axis ticks as percentage of the plot radius.
    tick_step : int, optional
        The interval between the displayed sequence positions.
        For example ``tick_step=200`` means that ticks will be displayed
        at ``1``, ``200``, ``400``, ``600``, etc.
    ring_width : float, optional
        The width of the outer ring as percentage of the plot radius.
    feature_width : float, optional
        The width of each feature arrow/arc as percentage of the
        plot radius.
    spacing : float, optional
        The spacing between the rows of feature arrows/arcs as
        percentage of the plot radius.
    arrow_head_length : float, optional
        The length of the arrow heads as percentage of the radius
        at which the respective arrow is plotted.
    label : str, optional
        The central label of the plot. Ususally the plasmid name.
    face_properties : dict, optional
        A dictionary of properties given to the patches that make up the
        feature arrows/arcs.
        Internally the arrow tail is a *Matplotlib* :class:`Rectangle`,
        and the arrow head is a :class:`Polygon`.
        For example this parameter could be used to add edge lines to
        the arrows/arcs.
    label_properties : dict, optional
        A dictionary of properties given to the feature labels.
        Internally each feature label is one or multiple *Matplotlib*
        :class:`Text` instances.
        For example this parameter could be used to set the font weight
        or size.
    omit_oversized_labels : bool, optional
        If true, a feature label will not be displayed, if the label is
        larger than its containing arc/arrow.
        This ensures, that labels from different features will not
        overlap each other.
        If false, the feature labels will always be displayed.
    feature_formatter : function, optional
        A function that determines *how* each feature is displayed.
        The given function must take a :class:`Feature` and must return
        the following tuple:

        - *directional* : bool
           
            True, if the direction of the feature should be indicated by
            an arrow.
            Otherwise, the feature is plotted is arc.
        
        - *face_color* : tuple or str, optional
            
            A *Matplotlib* compatible color for the feature arrow/arc.
        
        - *label_color* : tuple or str, optional
            
            A *Matplotlib* compatible color for the feature label.
        
        - *label* : str or None
            
            The label to be displayed for this feature.
            None, if no label should be displayed.
    """
    from matplotlib.projections.polar import PolarAxes
    
    if not isinstance(axes, PolarAxes):
        raise TypeError("The given axes must be a 'PolarAxes'")

    ### Setup parameters ###
    if plasmid_size is None:
        # 'stop' of 'get_location_range()' is exclusive -> -1
        plasmid_size = annotation.get_location_range()[1] - 1
    if face_properties is None:
        face_properties = {}
    if label_properties is None:
        label_properties = {}
    if feature_formatter is None:
        feature_formatter = _default_feature_formatter


    ### Setup matplotlib ###
    # The x-coordinate is given as angle (rad)
    # Full circle -> 2*pi
    axes.set_xlim(0, 2*np.pi)
    axes.set_ylim(0, 1)
    axes.yaxis.set_visible(False)
    axes.xaxis.set_tick_params(
        bottom=False, labelbottom=True
    )
    axes.set_theta_zero_location("N")
    axes.set_theta_direction("clockwise")
    axes.spines["polar"].set_visible(False)
    axes.grid(False)
    # Setup ticks
    ticks = [1]
    tick_labels = [str(1)]
    for tick in range(tick_step, plasmid_size, tick_step):
        ticks.append(tick)
        tick_labels.append(str(tick))
    # Sequence location is replaced by angle
    axes.xaxis.set_ticks([_loc_to_rad(tick, plasmid_size) for tick in ticks])
    axes.xaxis.set_ticklabels(tick_labels)
    ### Draw plasmid ring with ticks and central label ###
    
    # Plasmid ring
    # Use 'barh()' instead of a Rectangle patch to ensure that the axes
    # is properly initialized
    # Otherwise the feature rectangles are not curved, but straight
    axes.barh(
        1-ring_width-tick_length, 2*np.pi, ring_width,
        align="edge", color="black"
    )
    
    # Ticks (ticks itself, not the tick labels)
    for tick in ticks:
        angle = _loc_to_rad(tick, plasmid_size)
        axes.plot(
            (angle, angle), (1-tick_length, 1),
            color="black", linewidth=1, linestyle="-"
        )
    
    # Central plasmid label
    if label is not None:
        axes.text(
            0, 0, label, ha="center", va="center",
            color="black", size=32, fontweight="bold"
        )


    ### Draw plasmid interior ###
    inner_radius = 1 - ring_width - tick_length
    features = sorted(
        [
            _merge_over_periodic_boundary(feature, plasmid_size)
            for feature in annotation
        ],
        # Features are sorted by the length of their location range
        # The shortest come first
        key = lambda feature: np.diff(feature.get_location_range())[0],
        reverse = True
    )
    axes.add_artist(PlasmidMap(
        axes, 0, features, plasmid_size, inner_radius, feature_width, spacing,
        arrow_head_length, label, face_properties, label_properties,
        omit_oversized_labels, feature_formatter
    ))


try:
    # Only create these classes when matplotlib is installed
    from matplotlib.artist import Artist
    from matplotlib.transforms import Bbox
    from matplotlib.patches import Rectangle, Polygon


    class PlasmidMap(Artist):
        def __init__(self, axes, zorder, features, plasmid_size, radius,
                     feature_width, spacing, arrow_head_length, label,
                     face_properties, label_properties, omit_oversized_labels,
                     feature_formatter):
            super().__init__()
            self._axes = axes
            self.zorder = zorder
            self._features = features
            self._plasmid_size = plasmid_size
            self._radius = radius
            self._feature_width = feature_width
            self._spacing = spacing

            self._all_indicators = []
            for feature in features:
                indicators_for_feature = []
                for loc in feature.locs:
                     # Set proper positions in 'draw()' method
                    bbox = Bbox.from_extents(0, 0, 0, 0)
                    # Draw features as curved arrows (feature indicator)
                    indicator = axes.add_artist(Feature_Indicator(
                        axes, self.zorder + 1, feature, loc, bbox,
                        arrow_head_length, face_properties, label_properties,
                        omit_oversized_labels, feature_formatter
                    ))
                    indicators_for_feature.append(indicator)
                self._all_indicators.append(indicators_for_feature)

        
        def draw(self, renderer, *args, **kwargs):
            # Find the maximum amount of feature rows
            # (used for overlapping features)
            row_count = int(
                self._radius // (self._feature_width + self._spacing)
            )
            # Tracks the location ranges of feature that were added to
            # a row in order to check if that row is occupied
            ranges_in_row = [[] for i in range(row_count)]
            # Stores the bottom coordinate (radius) for each row
            row_bottoms = [
                self._radius - (row+1) * (self._feature_width + self._spacing)
                for row in range(row_count)
            ]

            # Arrange the feature indicators in an way,
            # that there is no overlap between them
            for feature, indicators_for_feature in zip(
                self._features, self._all_indicators
            ):
                row_bottom = None
                first, last = feature.get_location_range()

                for row_i, curr_range in enumerate(ranges_in_row):
                    is_occupied = False
                    if curr_range is not None:
                        # Check if row is occupied
                        for curr_first, curr_last in curr_range:
                            # If the location extends over periodic
                            # boundary the 'first' location is negative
                            if first > 0:
                                # 'Normal feature'
                                if first <= curr_last and last >= curr_first:
                                    is_occupied = True
                            else: # first < 1
                                # Location is over periodic boundary
                                if first + self._plasmid_size <= curr_last \
                                   or last >= curr_first:
                                        is_occupied = True
                    if not is_occupied:
                        # Row is not occupied by another feature
                        # in the location range of the new feature
                        # -> Use this row
                        if first > 0:
                            # 'Normal feature'
                            ranges_in_row[row_i].append((first, last))
                        else:
                            # Location is over periodic boundary
                            # Split into 'end' and 'start' part
                            ranges_in_row[row_i].append((
                                first + self._plasmid_size, self._plasmid_size
                            ))
                            ranges_in_row[row_i].append((
                                1, last
                            ))
                        row_bottom = row_bottoms[row_i]
                        break
                if row_bottom is None:
                    # No free row -> ignore feature and raise warning
                    warnings.warn(
                        "Too many feature overlaps, try to increase the "
                        "radius or decrease the feature width or spacing"
                    )
                else:
                    for loc, indicator in zip(
                        feature.locs, indicators_for_feature
                    ):
                        # Calculate arrow shape parameters
                        row_center = row_bottom + self._feature_width/2
                        row_top = row_bottom + self._feature_width
                        start_ang = _loc_to_rad(loc.first, self._plasmid_size)
                        stop_ang  = _loc_to_rad(loc.last, self._plasmid_size)
                        bbox = Bbox.from_extents(
                            start_ang, row_bottom, stop_ang, row_top
                        )
                        indicator.set_bbox(bbox)


    class Feature_Indicator(Artist):
        def __init__(self, axes, zorder, feature, loc, bbox, head_length,
                     arrow_properties, label_properties, omit_oversized_labels,
                     feature_formatter):
            super().__init__()
            self._axes = axes
            self.zorder = zorder
            self._direction = loc.strand
            self._bbox = bbox
            self._head_length = head_length
            self._omit_oversized_labels = omit_oversized_labels
            
            # Determine how to draw the feature
            directional, face_color, label_color, label \
                = feature_formatter(feature)
            
            # Draw arrow as composition of a rectangle and a triangle,
            # as FancyArrow does not properly work for polar plots

            self._arrow_tail = axes.add_patch(Rectangle(
                # Set positions in 'draw()' method
                (0, 0), 0, 0,
                # Line width is set to 1 to avoid strange artifact in
                # the transition from rectangle (tail) to polygon (head)
                color=face_color, linewidth=1, zorder = self.zorder + 1,
                **arrow_properties
            ))
            
            if directional:
                # Only draw any arrow head when feature has a direction,
                # otherwise simply draw the tail (rectangle)
                self._arrow_head = axes.add_patch(Polygon(
                    # Set positions in 'draw()' method
                    [(0, 0), (0, 0), (0, 0)],
                    color=face_color, linewidth=1, zorder = self.zorder + 1,
                    **arrow_properties
                ))
            else:
                self._arrow_head = None

            if label is not None:
                label_properties["color"] = label_color
                self._label = axes.add_artist(CurvedText(
                    # Set positions in 'draw()' method
                    axes, self.zorder + 1, 0, 0, label, label_properties
                ))
            else:
                self._label = None
        

        def set_bbox(self, bbox):
            self._bbox = bbox
            center_x = (bbox.x0 + bbox.x1) / 2
            center_y = (bbox.y0 + bbox.y1) / 2
            if self._label is not None:
                self._label.set_position(center_x, center_y)


        def draw(self, renderer, *args, **kwargs):
            bbox = self._bbox
            center_x = (bbox.x0 + bbox.x1) / 2
            center_y = (bbox.y0 + bbox.y1) / 2

            # Constant absolute width for all arrows
            # irrespective of the radius in the polar plot
            # Calculate actual angle from given absolute width
            head_length = self._head_length / center_y
            
            # Check if the head should be drawn
            if self._arrow_head is None:
                head_length = 0
            # Check if the feature location is too small for
            elif head_length > bbox.width:
                # Limit size of arrow head to range of location
                head_length = bbox.width

            if self._direction == Location.Strand.FORWARD:
                rect_pos = (bbox.x0, bbox.y0)
                # (x0, y0), (x1, y1), (x2, y2)
                triangle_coord = [
                    (bbox.x1 - head_length, bbox.y0), # base 1
                    (bbox.x1 - head_length, bbox.y1), # base 2
                    (bbox.x1,              center_y) # tip
                ]
            else:
                rect_pos = (bbox.x0+head_length, bbox.y0)
                triangle_coord = [
                    (bbox.x0 + head_length, bbox.y0), # base 1
                    (bbox.x0 + head_length, bbox.y1), # base 2
                    (bbox.x0,              center_y) # tip
                ]
            
            # Update coordinates of sub-artists
            self._arrow_tail.set_xy(rect_pos)
            self._arrow_tail.set_width(bbox.width-head_length)
            self._arrow_tail.set_height(bbox.height)
            if self._arrow_head is not None:
                self._arrow_head.set_xy(triangle_coord)
            
            if self._label is not None:
                # Do not draw the labels if it is larger than the
                # indicator
                if self._omit_oversized_labels \
                   and self._label.get_total_angle(renderer) > bbox.width:
                        self._label.set_visible(False)
                else:
                    self._label.set_visible(True)



    class CurvedText(Artist):
        def __init__(self, axes, zorder, angle, radius, string,
                     text_properties):
            super().__init__()
            self._axes = axes
            self.zorder = zorder
            self._angle = angle
            self._radius = radius

            self._texts = []
            for word in _split_into_words(string):
                text = axes.text(
                    # Set position in 'draw()' method
                    0, 0,
                    word,
                    ha="center", va="center",
                    zorder=self.zorder + 1,
                    **text_properties,
                )
                self._texts.append(text)
        

        def set_visible(self, visible):
            super().set_visible(visible)
            for text in self._texts:
                text.set_visible(visible)


        def set_position(self, angle, radius):
            self._angle = angle
            self._radius = radius
        

        def get_total_angle(self, renderer):
            return np.sum(self.get_word_angles(renderer))
        

        def get_word_angles(self, renderer):
            ax_px_radius = self._axes.get_window_extent(renderer).width / 2
            ax_unit_radius = self._axes.get_ylim()[1]
            circle_px_circumference = ax_px_radius * 2*np.pi \
                                      * (self._radius / ax_unit_radius)

            rad_angle = 360 - np.rad2deg(self._angle)
            # Avoid to draw the text upside down, when drawn on the
            # bottom half of the map
            if rad_angle > 90 and rad_angle < 270:
                turn_around = True
            else:
                turn_around = False
            
            angles = []
            for text in self._texts:
                orig_rot = text.get_rotation()
                orig_visible = text.get_visible()
                # Reset rotation and visibility
                # for correct window extent calculation
                text.set_rotation(0)
                text.set_visible(True)
                word_px_width = text.get_window_extent(renderer).width
                # In some Matplotlib versions the window extent of
                # whitespace characters is 'nan'
                # In this case, assign a fixed width
                if np.isnan(word_px_width):
                    word_px_width = 5.0
                word_angle \
                    = 2*np.pi * word_px_width / circle_px_circumference
                angles.append(word_angle)
                # Restore
                text.set_rotation(orig_rot)
                text.set_visible(orig_visible)
            return angles
        

        def draw(self, renderer, *args, **kwargs):
            angles = self.get_word_angles(renderer)
            total_angle = np.sum(angles)

            rad_angle = 360 - np.rad2deg(self._angle)
            # Avoid to draw the text upside down, when drawn on the
            # bottom half of the map
            if rad_angle > 90 and rad_angle < 270:
                turn_around = True
            else:
                turn_around = False
            
            # Now that the angle for each word is known,
            # the appropriate position and rotation can be set
            if turn_around:
                # curr_angle is the left-aligned position of the
                # upcoming word
                curr_angle = self._angle + total_angle / 2
            else:
                curr_angle = self._angle - total_angle / 2
            for text, angle in zip(self._texts, angles):
                if turn_around:
                    # The text itself is centered
                    # -> The position itself must be corrected with
                    # half of the word angle
                    angle_corrected = curr_angle - angle / 2
                    text_rot = 360 - np.rad2deg(angle_corrected) + 180
                    curr_angle -= angle
                else:
                    angle_corrected = curr_angle + angle / 2
                    text_rot = 360 - np.rad2deg(angle_corrected)
                    curr_angle += angle
                text.set_position((angle_corrected, self._radius))
                text.set_rotation(text_rot)


except ImportError:
    pass




def _loc_to_rad(loc, plasmid_size):
    if loc > plasmid_size:
        raise ValueError(
            f"Location {loc} is larger then the plasmid size of {plasmid_size}"
        )
    # Location starts at 1 -> (loc-1)
    return ((loc-1) / plasmid_size) * 2*np.pi


def _rad_to_loc(rad, plasmid_size):
    # Location starts at 1 -> + 1
    return rad / (2*np.pi) * plasmid_size + 1


def _merge_over_periodic_boundary(feature, plasmid_size):
    if len(feature.locs) == 1:
        # Only one location -> no merge possible
        return feature
    first_loc = None
    last_loc  = None
    # Find total first location of the feature
    for loc in feature.locs:
        if first_loc is None or loc.first < first_loc.first:
            first_loc = loc
    # Find total last location of the feature
    for loc in feature.locs:
        if last_loc is None or loc.last > last_loc.last:
            last_loc = loc
    # If the first and last location meet at the periodic boundary of
    # the plasmid -> merge them
    if first_loc.first == 1 and last_loc.last == plasmid_size \
        and first_loc.strand == last_loc.strand:
            new_locs = set(feature.locs)
            new_locs.remove(first_loc)
            new_locs.remove(last_loc)
            new_locs.add(Location(
                # the fist base is now at negative location
                # by shifting by one plasmid 'period'
                first  = last_loc.first - plasmid_size,
                last   = first_loc.last,
                strand = first_loc.strand,
                defect = first_loc.defect | last_loc.defect
            ))
            return Feature(feature.key, new_locs, feature.qual)
    else:
        return feature


# ' ', '-' and '_' are word delimiters
separators = re.compile(r"\s|_|-")
def _split_into_words(string):
    match_indices = sorted(
        [match.start() for match in separators.finditer(string)]
    )
    current_index = 0
    words = []
    for i in match_indices:
        # Add word up to delimiter
        words.append(string[current_index : i])
        # Add delimiter
        words.append(string[i : i+1])
        current_index = i+1
    # If there is a word after the last delimiter, add it too
    if current_index < len(string):
        words.append(string[current_index:])
    return words


def _default_feature_formatter(f):
    """
    Returns
    -------
    directional : bool
        True, if the direction of the feature should be indicated by
        an arrow.
    face_color: tuple or str, optional
        A matplotlib compatible color for the feature indicator.
    label_color: tuple or str, optional
        A matplotlib compatible color for the feature label.
    label: str or None
        The label to be displayed for this feature.
        None, if no label should be displayed.
    """
    # Source
    if f.key == "source":
        if f.qual.get("organism") is not None:
            label = f"Source: {f.qual.get('organism')}"
        else:
            label = None
        return False, "black", "white", label
    
    # Origin of Replication
    elif f.key == "rep_origin":
        return False, "indigo", "white", \
               f.qual.get("standard_name", "ori")
    
    # Coding sequences
    elif f.key in ["gene", "CDS", "rRNA"]:
        label = f.qual.get("product")
        if label is None:
            label = f.qual.get("gene")
        return True, colors["orange"], "black", label
    
    elif f.key == "regulatory":
        # Promoters
        if f.qual.get("regulatory_class") in [
            "promoter",
            "TATA_box",
            "minus_35_signal",
            "minus_10_signal"
        ]:
            return True, colors["dimgreen"], "black", f.qual.get("note")
        
        # Terminators
        elif f.qual.get("regulatory_class") in "terminator":
            return False, "firebrick", "white", f.qual.get("note")
        
        # RBS
        elif f.qual.get("regulatory_class") == "ribosome_binding_site":
            return False, colors["brightorange"], "white", None
    
    # Primers
    elif f.key == "primer_bind":
        return True, "royalblue", "black", f.qual.get("note")
    
    # Binding proteins
    elif f.key == "protein_bind":
        return False, colors["lightgreen"], "black", f.qual.get("note")
    
    # Misc
    return True, "dimgray", "white", f.qual.get("note")