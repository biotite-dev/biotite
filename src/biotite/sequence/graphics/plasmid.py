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
                     label_properties=None):
    from matplotlib.patches import Rectangle, Polygon
    
    ### Setup parameters ###
    if loc_range is None:
        loc_range = annotation.get_location_range()
    else:
        annotation = annotation[loc_range[0] : loc_range[1]]
    if label_properties is None:
         label_properties = {}
    
    
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

    
    ### Draw plasmid ring with ticks ###
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
            head_angle_width = arrow_head_width / row_center
            start_angle = _loc_to_rad(loc.first, loc_range)
            stop_angle  = _loc_to_rad(loc.last, loc_range)
            center_angle = start_angle + stop_angle/2
            # Check if the feature location is too small for
            # arrow tail AND head to be drawn
            if head_angle_width > stop_angle - start_angle:
                draw_rect = False
                # Limit size of arrow head to range of location
                head_angle_width = stop_angle - start_angle
            else:
                draw_rect = True
            # Draw arrow as compostion of a rectangle and a triangle
            # FancyArrow does not properly work for polar plots
            if draw_rect:
                # Linewidth is set to 1 to avoid strange artifact
                # in the transition from rectangle to polygon
                axes.add_patch(Rectangle(
                    (start_angle, row_bottom),
                    stop_angle-start_angle-head_angle_width, feature_width,
                    color=colors["dimorange"], linewidth=1
                ))
            axes.add_patch(Polygon(
                [
                    (stop_angle-head_angle_width, row_bottom),
                    (stop_angle-head_angle_width, row_top),
                    (stop_angle,                  row_center)
                ],
                color=colors["dimorange"], linewidth=1
            ))
            # Draw feature label
            center_angle = (start_angle + stop_angle) / 2
            text_rot = 360 - np.rad2deg(center_angle)
            # Do not draw text upside down
            if text_rot > 90 and text_rot < 270:
                text_rot += 180
            axes.text(
                center_angle, row_center, "text",
                ha="center", va="center",
                rotation=text_rot,
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
