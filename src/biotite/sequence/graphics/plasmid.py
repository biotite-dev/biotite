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


def plot_plasmid_map(axes, annotation, loc_range=None,
                     feature_width=1.0, ring_width=0.2, spacing=0.2,
                     radius=15, arrow_head_width=0.5, label_properties=None):
    from matplotlib.patches import Rectangle, Polygon
    
    ### Setup parameters ###
    if loc_range is None:
        loc_range = annotation.get_location_range()
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
    #axes.grid(False)

    
    ### Draw plasmid ring ###
    axes.barh(
        radius-ring_width, 2*np.pi, ring_width,
        align="edge", color="black"
    )

    
    ### Draw features ###
    # Find the maximum amount of feature rows
    # (used for overlapping features)
    radius_eff = radius - ring_width
    row_count = int(radius_eff // (feature_width + spacing))
    # Draw features as curved arrows
    # Tracks the last feature's location range that was added to a row
    # in order to check if that row is currently occupied
    current_ranges = [None] * row_count
    # Stores the bottom coordinate (radius) for each row
    row_bottoms = [radius_eff - (row+1) * (feature_width+spacing)
                  for row in range(row_count)]
    features = sorted(list(annotation))

    for feature in features:
        row_bottom = None
        first, last = feature.get_location_range()
        for row_i, curr_range in enumerate(current_ranges):
            if curr_range is None or first > curr_range[1]:
                # Row is not occupied by another feature
                # in the location range of the new feature
                # -> Use this row
                current_ranges[row_i] = (first, last)
                row_bottom = row_bottoms[row_i]
                break
        if row_bottom is None:
            # No free row
            raise ValueError(
                "Too many feature overlaps, try to increase the radius "
                "or decrease the feature width or spacing"
            )
        for loc in feature.locs:
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