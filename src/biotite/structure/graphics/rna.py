# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.graphics"
__author__ = "Tom David Müller"
__all__ = ["plot_nucleotide_secondary_structure"]

import shutil
import numpy as np
import matplotlib.pyplot as plt
from .. import pseudoknots
from ...application.viennarna import RNAplotApp


def plot_nucleotide_secondary_structure(
    base_labels, base_pairs, length, layout_type=RNAplotApp.Layout.NAVIEW, 
    draw_pseudoknots=True, pseudoknot_order=None, angle=0, bond_linewidth=1, 
    bond_linestyle=None, bond_color='black', backbone_linewidth=1, 
    backbone_linestyle='solid', backbone_color='grey', 
    base_font={'size': 'small'}, base_box={'pad': 0, 'color': 'white'}, 
    annotation_positions=None, annotation_offset=8.5, 
    annotation_font={'size': 'smaller'}, bin_path="RNAplot"
    ):
    """
    Generate 2D plots of nucleic acid secondary structures using the 
    interface to *RNAplot*, which is part of the *ViennaRNA* software
    package.

    Internally a :class:`RNAplotAPP` instance is created to generate 
    coordinates for each individual base on a 2D plane. *ViennaRNA* must 
    be installed in order to use this function.

    Parameters
    ----------
    base_labels : iterable
        The labels denoting the type of each base.
    base_pairs : ndarray, shape=(n,2)
        Each row corresponds to the positions of the bases in the
        sequence. The positions are counted from zero.
    length : int
        The number of bases in the sequence.
    layout_type : RNAplotApp.Layout (default: RNAplotApp.Layout.NAVIEW)
        The layout type according to the *RNAplot* documentation.
    draw_pseudoknots : bool, optional (default: True)
        Whether pseudoknotted bonds should be drawn.
    angle : int
        The angle the plot should be rotated.
    bond_linewidth : float or ndarray, shape(n,), optional (default: 1)
        The linewidth of each bond. Provide a single value to set the
        linewidth for all bonds or an array to set the linewidth for 
        each individual bond.
    bond_linestyle : str or ndarray, shape(n,), optional (default: None)
        The *Matplotlib* compatible linestyle of each bond. Provide a
        single value to set the linewidth for all bonds or an array to 
        set the linewidth for each individual bond. By default, solid 
        lines are used for non-pseudoknotted bonds and dashed lines are 
        used for pseudoknotted bonds.
    bond_color : str or ndarray, shape(n,) or shape(n,3) or shape(n,4),
                 optional (default: 'black')
        The *Matplotlib* compatible color of each bond. Provide a single 
        string to set the color for all bonds or an array to set the 
        color for each individual bond. Furthermore, the color for each 
        individual bond can be provided as array of RGB or RGBA colors.
    backbone_linewidth : float, optional (default: 1)
        The linewidth of the backbone.
    backbone_linestyle : str (default: 'solid')
        The *Matplotlib* compatible linestyle of the backbone.
    backbone_color : str or ndarray, shape=(3,) or shape=(4,), 
                     dtype=float, optional (default: 'grey')
        The *Matplotlib* compatible color of the backbone.
    base_font : dict, optional (default: {'size': 'smaller'})
        The *Matplotlib* compatible font of the labels denoting the type
        of each base.
    base_box : dict or ndarray, shape(n,), optional 
               (default: dict(pad=0, color='white'))
        The *Matplotlib* compatible properties of the ``FancyBboxPatch``
        surrounding the base labels. Provide a single dictionary to
        set the properties of all base lables or an array to set the
        properties for each individual label.
    annotation_positions : iterable, optional (default: None)
        The positions of the bases to be numbered. By default every 
        second base is annotated. Please note that while the positions 
        in the sequence are counted from zero, they are displayed on the 
        graph counted from one.
    annotation_offset : int, optional (default: 8.5)
        The offset of the annotations from the base labels.
    annotation_font : dict, optional (default: {'size': 'smaller'})
        The *Matplotlib* compatible font of the annotations.
    bin_path : str, optional
        Path of the *RNAplot* binary.
    """

    # Check if RNAplot is installed
    if shutil.which(bin_path) is None:
        raise FileNotFoundError(
            'RNAplot is not installed at the specified location, unable to '
            'plot secondary structure.'
        )

    # Get the unknotted base pairs
    if pseudoknot_order is None:
        pseudoknot_order = pseudoknots(base_pairs, max_pseudoknot_order=0)[0]
    unknotted_base_pairs = base_pairs[pseudoknot_order == 0]

    # If `bond_linewidth` is not an array, extrapolate
    if not isinstance(bond_linewidth, np.ndarray):
        bond_linewidth = np.full(base_pairs.shape[0], bond_linewidth)

    # If `bond_color` is not an array, extrapolate
    if not isinstance(bond_color, np.ndarray):
        bond_color = np.full(base_pairs.shape[0], bond_color)

    # If `background_color` is not an array, extrapolate
    if not isinstance(base_box, np.ndarray):
        base_box = np.full(length, base_box)

    # By default pseudoknotted bonds are denoted as dashed lines, while
    # unknotted bonds are denoted as solid lines
    if bond_linestyle is None:
        bond_linestyle = np.full(base_pairs.shape[0], 'solid', dtype='object')
        bond_linestyle[pseudoknot_order != 0] = 'dashed'
    # If `bond_linestyle` is not an array, extrapolate
    elif not isinstance(bond_linestyle, np.ndarray):
        bond_linestyle = np.full(base_pairs.shape[0], bond_linestyle)

    # If pseudoknots are not to be drawn, remove pseudoknotted bonds, 
    # regardless of the given linestyles
    if not draw_pseudoknots:
        bond_linestyle[pseudoknot_order != 0] = 'None'
    
    
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
    for coords, label, box in zip(coordinates, base_labels, base_box):
        t = ax.text(
                    x=coords[0], y=coords[1], s=label, 
                    font=base_font, ha='center', va='center'
        )
        t.set_bbox(box)

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
            ha='center', va='center', font=annotation_font
        )