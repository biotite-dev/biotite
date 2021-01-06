# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.graphics"
__author__ = "Tom David Müller"
__all__ = ["plot_nucleotide_secondary_structure"]

import numpy as np
import matplotlib.pyplot as plt
from .. import pseudoknots
from ...application.viennarna import RNAplotApp


def plot_nucleotide_secondary_structure(
    base_labels, base_pairs, length, layout_type=1, pseudoknot_order=None, 
    angle=0, bond_linewidth=1, bond_linestyle=None, bond_color='black',
    backbone_linewidth=1, backbone_linestyle='solid', backbone_color='grey', 
    base_fontsize='smaller', annotation_positions=None, annotation_offset=8.5, 
    annotation_fontsize='smaller', bin_path="RNAplot"
    ):

    #TODO: Check if RNAplot is installed

    # Get the unknotted base pairs
    if pseudoknot_order is None:
        pseudoknot_order = pseudoknots(base_pairs)[0]
    unknotted_base_pairs = base_pairs[pseudoknot_order == 0]

    # If `bond_linewidth` is not an array, extrapolate
    if not isinstance(bond_linewidth, np.ndarray):
        bond_linewidth = np.full(base_pairs.shape[0], bond_linewidth)

    # If `bond_color` is not an array, extrapolate
    if not isinstance(bond_color, np.ndarray):
        bond_color = np.full(base_pairs.shape[0], bond_color)

    # By default pseudoknotted bonds are denoted as dashed lines, while
    # unknotted bonds are denoted as solid lines
    if bond_linestyle is None:
        bond_linestyle = np.full(base_pairs.shape[0], 'solid', dtype='object')
        bond_linestyle[pseudoknot_order != 0] = 'dashed'
    # If `bond_linestyle` is not an array, extrapolate
    elif not isinstance(bond_linestyle, np.ndarray):
        bond_linestyle = np.full(base_pairs.shape[0], bond_linestyle)
    
    
    # If no specific annotation positions are given, annotate every
    # second base pair
    if annotation_positions is None:
        annotation_positions = range(0, length, 2)

    # Get coordinates for secondary structure plot
    coordinates = RNAplotApp.compute_coordinates(
    base_pairs=unknotted_base_pairs, 
    length=length,
    bin_path=bin_path,
    layout_type=layout_type
    )

    # Rotate Coordinates
    if angle != 0:
        angle = np.deg2rad(angle)
        R = np.array(
            [[np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]]
        )
        for i, coord in enumerate(coordinates):
            coordinates[i] = np.dot(R, coord)

    # Create Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Remove axes and frame
    ax.set_xticks([])
    ax.set_yticks([])
    plt.box(False)

    # Define buffer area (Border)
    coord_range = abs(np.max(coordinates)) + abs(np.min(coordinates))
    buffer = 0.03*coord_range

    # Adjust display
    ax.set_xlim(
        np.min(coordinates[:,0])-buffer, np.max(coordinates[:,0])+buffer
    )
    ax.set_ylim(
        np.min(coordinates[:,1])-buffer, np.max(coordinates[:,1])+buffer
    )
    ax.set_aspect(aspect='equal')

    # Draw backbone
    ax.plot(coordinates[:,0], coordinates[:,1], color=backbone_color, 
            linestyle=backbone_linestyle, linewidth=backbone_linewidth)

    # Draw base labels
    for coords, label in zip(coordinates, base_labels):
        t = ax.text(
                    x=coords[0], y=coords[1], s=label, 
                    fontsize=base_fontsize, ha='center', va='center'
        )
        t.set_bbox(dict(pad=0, color='white'))

    # Draw bonds
    for (base1, base2), color, style, width in zip(
        base_pairs, bond_color, bond_linestyle, bond_linewidth
    ):
        base1_coords = coordinates[base1]
        base2_coords = coordinates[base2]
        x = base1_coords[0], base2_coords[0]
        y = base1_coords[1], base2_coords[1]
        ax.plot(x, y, color=color, linestyle=style, linewidth=width)

    # Draw annotations
    for i in annotation_positions:
        if (i > 0) and ((i+1) < length):
            # Get the average of the direction vectors to the next and
            # previous base
            vector_to_previous = np.array(
                [coordinates[i-1][0] - coordinates[i][0],
                 coordinates[i-1][1] - coordinates[i][1]]
            )
            vector_to_previous = vector_to_previous / np.linalg.norm(
                vector_to_previous
            )
            vector_to_next = np.array(
                [coordinates[i][0] - coordinates[i+1][0],
                 coordinates[i][1] - coordinates[i+1][1]]
            )
            vector_to_next = vector_to_next / np.linalg.norm(
                vector_to_next
            )
            vector = (vector_to_next + vector_to_previous) / 2
        elif i > 0:
            # For the last base get the direction vector to the previous
            # base
            vector = np.array(
                [coordinates[i-1][0] - coordinates[i][0],
                 coordinates[i-1][1] - coordinates[i][1]]
            )
        else:
            # For the first base get the direction vector to the next
            # base
            vector = np.array(
                [coordinates[i][0] - coordinates[i+1][0],
                 coordinates[i][1] - coordinates[i+1][1]]
            )
        # Normalize the vector
        vector = vector / np.linalg.norm(vector)
        # Get the perpendicular vector
        vector = np.array([vector[1], -vector[0]])
        # The annotations are offset in the direction of the 
        # perpendicular vector
        x, y = coordinates[i] + (annotation_offset*vector)
        ax.text(
            x=x, y=y, s=i+1, 
            ha='center', va='center', fontsize=annotation_fontsize
        )