# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["plot_plasmid_map"]

import copy
import abc
import numpy as np
from ...visualize import colors
from ..annotation import Annotation, Feature, Location


def plot_plasmid_map(axes, annotation, loc_range=None, radius=15,
                     tick_length= 0.2, tick_step=200, ring_width=0.2,
                     feature_width=1.0, spacing=0.2, arrow_head_width=0.5,
                     label=None, face_properties=None, label_properties=None,
                     feature_formatter=None):
    from matplotlib.transforms import Bbox
    
    ### Setup parameters ###
    if loc_range is None:
        loc_range = annotation.get_location_range()
    else:
        annotation = annotation[loc_range[0] : loc_range[1]]
    if face_properties is None:
         face_properties = {}
    if label_properties is None:
         label_properties = {}
    if feature_formatter is None:
        feature_formatter = _default_feature_formatter
    
    
    ### Setup matplotlib ###
    axes.set_xlim(0, 2*np.pi)
    axes.set_ylim(0, radius)
    axes.yaxis.set_visible(False)
    axes.xaxis.set_tick_params(
        bottom=False, labelbottom=True
    )
    axes.set_theta_zero_location("N")
    axes.set_theta_direction("clockwise")
    axes.spines["polar"].set_visible(False)
    axes.grid(False)
    # Setup ticks: Angle is replaced by sequence location
    ticks = [loc_range[0]] \
            + [(i+1) * tick_step for i in range(0, loc_range[1] // tick_step)]
    axes.xaxis.set_ticks([_loc_to_rad(tick, loc_range) for tick in ticks])
    axes.xaxis.set_ticklabels(ticks)

    
    ### Draw plasmid ring with ticks and central label ###
    axes.barh(
        radius-ring_width-tick_length, 2*np.pi, ring_width,
        align="edge", color="black"
    )
    for tick in ticks:
        angle = _loc_to_rad(tick, loc_range)
        axes.plot(
            (angle, angle), (radius-tick_length, radius),
            color="black", linewidth=1, linestyle="-"
        )
    if label is not None:
        axes.text(
            0, 0, label, ha="center", va="center",
            color="black", size=32, fontweight="bold"
        )

    
    ### Draw features ###
    # Find the maximum amount of feature rows
    # (used for overlapping features)
    radius_eff = radius - ring_width - tick_length
    row_count = int(radius_eff // (feature_width + spacing))
    # Tracks the location ranges of feature that were added to a row
    # in order to check if that row is occupied
    ranges_in_row = [[] for i in range(row_count)]
    # Stores the bottom coordinate (radius) for each row
    row_bottoms = [radius_eff - (row+1) * (feature_width+spacing)
                  for row in range(row_count)]
    features = sorted(
        [_merge_over_periodic_boundary(feature, loc_range)
         for feature in annotation],
        # Features are sorted by the length of their location range
        # The shortest come first
        key = lambda feature: np.diff(feature.get_location_range())[0],
        reverse = True
    )

    for feature in features:
        row_bottom = None
        first, last = feature.get_location_range()
        for row_i, curr_range in enumerate(ranges_in_row):
            is_occupied = False
            if curr_range is not None:
                # Check if row is occupied
                for curr_first, curr_last in curr_range:
                    if first <= curr_last and last >= curr_first:
                        is_occupied = True
            if not is_occupied:
                # Row is not occupied by another feature
                # in the location range of the new feature
                # -> Use this row
                ranges_in_row[row_i].append((first, last))
                row_bottom = row_bottoms[row_i]
                break
        if row_bottom is None:
            # No free row
            raise ValueError(
                "Too many feature overlaps, try to increase the radius "
                "or decrease the feature width or spacing"
            )
        for loc in feature.locs:
            # Draw features as curved arrows
            # Calculate arrow shape parameters
            row_center = row_bottom + feature_width/2
            row_top = row_bottom + feature_width
            # Same apparent width for all arrows,
            # irrespective of the radius of the polar plot
            head_angle_width = arrow_head_width / row_center
            start_angle = _loc_to_rad(loc.first, loc_range)
            stop_angle  = _loc_to_rad(loc.last, loc_range)
            bbox = Bbox.from_extents(
                start_angle, row_bottom, stop_angle, row_top
            )
            _draw_location(
                axes, feature, loc, bbox, head_angle_width,
                face_properties, label_properties, feature_formatter
            )
            

def _draw_location(axes, feature, loc, bbox, head_width,
                   face_properties, label_properties, feature_formatter):
    from matplotlib.patches import Rectangle, Polygon
    
    # Determine how to draw the feature
    directional, face_color, text_color, text = feature_formatter(feature)

    center_x = (bbox.x0 + bbox.x1) / 2
    center_y = (bbox.y0 + bbox.y1) / 2
    # Check if the feature location is too small for
    # arrow tail AND head to be drawn
    if not directional:
        draw_rect = True
        draw_head = False
        head_width = 0
    elif head_width > bbox.width:
        draw_rect = False
        draw_head = True
        # Limit size of arrow head to range of location
        head_width = bbox.width
    else:
        draw_rect = True
        draw_head = True
    if loc.strand == Location.Strand.FORWARD:
        # (x0, y0), width, height
        rect_coord = (bbox.p0, bbox.width-head_width, bbox.height)
        # (x0, y0), (x1, y1), (x2, y2)
        triangle_coord = [
            (bbox.x1 - head_width, bbox.y0),
            (bbox.x1 - head_width, bbox.y1),
            (bbox.x1,              center_y)
        ]
    else:
        rect_coord = (bbox.p0+head_width, bbox.width-head_width, bbox.height)
        triangle_coord = [
            (bbox.x0 + head_width, bbox.y0),
            (bbox.x0 + head_width, bbox.y1),
            (bbox.x0,              center_y)
        ]
    # Draw arrow as composition of a rectangle and a triangle
    # FancyArrow does not properly work for polar plots
    if draw_rect:
        # Line width is set to 1 to avoid strange artifact
        # in the transition from rectangle to polygon
        axes.add_patch(Rectangle(
            *rect_coord,
            color=face_color, linewidth=1, **face_properties
        ))
    if draw_head:
        axes.add_patch(Polygon(
            triangle_coord,
            color=face_color, linewidth=1, **face_properties
        ))
    # Draw feature label
    text_rot = 360 - np.rad2deg(center_x)
    # Do not draw text upside down
    if text_rot > 90 and text_rot < 270:
        text_rot += 180
    if text is not None:
        axes.text(
            center_x, center_y, text,
            ha="center", va="center", rotation=text_rot, color=text_color,
            **label_properties
        )


def _loc_to_rad(loc, loc_range):
    start = loc_range[0]
    stop = loc_range[1]
    return (loc-start) / (stop-start) * 2*np.pi


def _rad_to_loc(rad, loc_range):
    start = loc_range[0]
    stop = loc_range[1]
    return rad / (2*np.pi) * (stop-start) + start


def _merge_over_periodic_boundary(feature, loc_range):
    if len(feature.locs) == 1:
        # Only one location -> no merge possible
        return feature
    first_loc = None
    last_loc  = None
    for loc in feature.locs:
        if first_loc is None or loc.first < first_loc.first:
            first_loc = loc
    for loc in feature.locs:
        if last_loc is None or loc.last > last_loc.last:
            last_loc = loc
    if first_loc.first == loc_range[0] and last_loc.last == loc_range[1]-1 \
        and first_loc.strand == last_loc.strand:
            new_locs = set(feature.locs)
            new_locs.remove(first_loc)
            new_locs.remove(last_loc)
            new_locs.add(Location(
                first  = loc_range[0] - (loc_range[1] - last_loc.first),
                last   = first_loc.last,
                strand = first_loc.strand,
                defect = first_loc.defect | last_loc.defect
            ))
            return Feature(feature.key, new_locs, feature.qual)
    else:
        return feature


def _default_feature_formatter(f):
    """
    Returns
    -------
    directional : bool
        True, if the direction of the feature should be indicated by
        an arrow.
    face_color: tuple or str, optional
        A matplotlib compatible color for the feature indicator.
    text_color: tuple or str, optional
        A matplotlib compatible color for the feature text.
    text: str or None
        The text to be displayed for this feature.
        None, if no text should be displayed.
    """
    # Source
    if f.key == "source":
        return False, colors["darkorange"], "white", f.qual.get("organism")
    
    # Origin of Replication
    elif f.key == "rep_origin":
        return False, colors["dimorange"], "black", \
               f.qual.get("standard_name", "ori")
    
    # Coding sequences
    elif f.key in ["gene", "CDS", "rRNA"]:
        return True, colors["darkgreen"], "black", f.qual.get("gene")
    
    elif f.key == "regulatory":
        # Promoters
        if f.qual.get("regulatory_class") in [
            "promoter",
            "TATA_box",
            "minus_35_signal",
            "minus_10_signal"
        ]:
            return True, colors["green"], "black", f.qual.get("gene")
        
        # Terminators
        elif f.qual.get("regulatory_class") in [
            "terminator"
        ]:
            return True, "black", "white", f.qual.get("gene")
    
    # Misc
    return True, colors["lightorange"], "black", f.qual.get("gene")