# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.graphics"
__author__ = "Tom David Müller"
__all__ = ["plot_nucleotide_secondary_structure"]

import shutil
from collections.abc import Iterable, Sequence
from itertools import repeat
from typing import Any
import numpy as np
from matplotlib.axes import Axes
from biotite.application.viennarna import RNAplotApp
from biotite.structure import pseudoknots
from biotite.typing import C2, K, MplColor, NDArray1, NDArray2


def plot_nucleotide_secondary_structure(
    axes: Axes,
    base_labels: Iterable[str],
    base_pairs: NDArray2[K, C2, np.integer],
    length: int,
    layout_type: RNAplotApp.Layout = RNAplotApp.Layout.NAVIEW,
    draw_pseudoknots: bool = True,
    pseudoknot_order: NDArray1[K, np.integer] | None = None,
    angle: float = 0,
    bond_linewidth: float | Iterable[float] = 1,
    bond_linestyle: str | Iterable[str] | None = None,
    bond_color: Sequence[MplColor] | MplColor = "black",
    backbone_linewidth: float = 1,
    backbone_linestyle: str = "solid",
    backbone_color: MplColor = "grey",
    base_text: dict[str, Any] | Iterable[dict[str, Any]] | None = None,
    base_box: dict[str, Any] | Iterable[dict[str, Any]] | None = None,
    annotation_positions: Iterable[int] | None = None,
    annotation_offset: float = 8.5,
    annotation_text: dict[str, Any] | Iterable[dict[str, Any]] | None = None,
    border: float = 0.03,
    bin_path: str = "RNAplot",
) -> None:
    """
    Generate 2D plots of nucleic acid secondary structures using the
    interface to *RNAplot*, which is part of the *ViennaRNA* software
    package.

    Internally a :class:`biotite.application.viennarna.RNAplotApp`
    instance is created to generate coordinates for each individual base
    on a 2D plane. *ViennaRNA* must be installed in order to use this
    function.

    Parameters
    ----------
    axes : Axes
        A *Matplotlib* axes, that is used as plotting area.
    base_labels : iterable
        The labels denoting the type of each base.
    base_pairs : ndarray, shape=(n,2), dtype=int
        Each row corresponds to the positions of the bases in the
        sequence. The positions are counted from zero.
    length : int
        The number of bases in the sequence.
    layout_type : RNAplotApp.Layout, optional
        The layout type according to the *RNAplot* documentation.
    draw_pseudoknots : bool, optional
        Whether pseudoknotted bonds should be drawn.
    pseudoknot_order : ndarray, shape=(k,), dtype=int, optional
        The pseudoknot order of each pair in the input `base_pairs`.
        If no pseudoknot order is given, a solution determined by
        :func:`biotite.structure.pseudoknots` is picked at random.
    angle : int or float, optional
        The angle the plot should be rotated.
    bond_linewidth : float or int or iterable, optional
        The linewidth of each bond. Provide a single value to set the
        linewidth for all bonds or an iterable to set the linewidth for
        each individual bond.
    bond_linestyle : str or iterable, optional
        The *Matplotlib* compatible linestyle of each bond. Provide a
        single value to set the linewidth for all bonds or an iterable
        to set the linewidth for each individual bond. By default, solid
        lines are used for non-pseudoknotted bonds and dashed lines are
        used for pseudoknotted bonds.
    bond_color : str or ndarray, shape(n,) or shape(n,3) or shape(n,4), optional
        The *Matplotlib* compatible color of each bond. Provide a single
        string to set the color for all bonds or an array to set the
        color for each individual bond.
    backbone_linewidth : float, optional
        The linewidth of the backbone.
    backbone_linestyle : str, optional
        The *Matplotlib* compatible linestyle of the backbone.
    backbone_color : str or ndarray, shape=(3,) or shape=(4,), dtype=float, optional
        The *Matplotlib* compatible color of the backbone.
    base_text : dict or iterable, optional
        The keyword parameters for the *Matplotlib* ``Text`` objects
        denoting the type of each base. Provide a single value to set
        the parameters for all labels or an iterable to set the
        parameters for each individual label.
        The default is ``{'size': 'small'}``.
    base_box : dict or iterable, optional)
        The *Matplotlib* compatible properties of the ``FancyBboxPatch``
        surrounding the base labels. Provide a single dictionary to
        set the properties of all base lables or an iterable to set the
        properties for each individual label.
        The default is ``{'pad'=0, 'color'='white'}``.
    annotation_positions : iterable, optional
        The positions of the bases to be numbered. By default every
        second base is annotated. Please note that while the positions
        in the sequence are counted from zero, they are displayed on the
        graph counted from one.
    annotation_offset : int or float, optional
        The offset of the annotations from the base labels.
    annotation_text : dict or iterable, optional
        The keyword parameters for the *Matplotlib* ``Text`` objects
        annotating the sequence. Provide a single value to set the
        parameters for all annotations or an iterable to set the
        parameters for each individual annotation.
        The default is ``{'size': 'small'}``.
    border : float, optional
        The percentage of the coordinate range to be left as whitespace
        to create a border around the plot.
    bin_path : str, optional
        Path to the *RNAplot* binary.
    """

    # Check if RNAplot is installed
    if shutil.which(bin_path) is None:
        raise FileNotFoundError(
            "RNAplot is not installed at the specified location, unable to "
            "plot secondary structure."
        )

    # Get the unknotted base pairs
    if pseudoknot_order is None:
        # Get a random unknotted structure retaining the maximum number
        # of base pairs.
        pseudoknot_order = pseudoknots(base_pairs, max_pseudoknot_order=0)[0]
    unknotted_base_pairs = base_pairs[pseudoknot_order == 0]

    # If `bond_linewidth` is a single value, extrapolate
    if isinstance(bond_linewidth, int) or isinstance(bond_linewidth, float):
        bond_linewidth = np.full(base_pairs.shape[0], bond_linewidth)

    # If `bond_color` is not an array, extrapolate
    if isinstance(bond_color, str) or not isinstance(bond_color, Sequence):
        bond_color = [bond_color] * base_pairs.shape[0]

    # Set the default properties of the Matplotlib `bbox` surrounding
    # the base labels
    base_box_iter: Iterable[dict[str, Any]]
    if base_box is None:
        base_box_iter = np.full(length, {"pad": 0, "color": "white"})
    elif isinstance(base_box, dict):
        base_box_iter = np.full(length, base_box)
    else:
        base_box_iter = base_box

    # By default pseudoknotted bonds are denoted as dashed lines, while
    # unknotted bonds are denoted as solid lines
    if bond_linestyle is None:
        bond_linestyle = np.full(base_pairs.shape[0], "solid", dtype="object")
        bond_linestyle[pseudoknot_order != 0] = "dashed"
    # If `bond_linestyle` is a string, extrapolate
    elif isinstance(bond_linestyle, str):
        bond_linestyle = np.full(base_pairs.shape[0], bond_linestyle, dtype="object")

    # If pseudoknots are not to be drawn, remove pseudoknotted bonds,
    # regardless of the given linestyles
    if not draw_pseudoknots:
        # Ensure that the array can hold the 'None' value
        # (not possible with 'U1' dtype for example)
        bond_linestyle = np.asarray(bond_linestyle, dtype="object")
        bond_linestyle[pseudoknot_order != 0] = "None"

    # Set the default properties of the base labels
    base_text_iter: Iterable[dict[str, Any]]
    if base_text is None:
        base_text_iter = np.full(length, {"size": "small"})
    elif isinstance(base_text, dict):
        base_text_iter = np.full(length, base_text)
    else:
        base_text_iter = base_text

    # If no specific annotation positions are given, annotate every
    # second base pair
    if annotation_positions is None:
        annotation_positions = range(0, length, 2)

    # Set the default font properties of the base annotations
    annotation_text_iter: Iterable[dict[str, Any]]
    if annotation_text is None:
        annotation_text_iter = repeat({"size": "small"})
    elif isinstance(annotation_text, dict):
        annotation_text_iter = repeat(annotation_text)  # pyright: ignore[reportAssignmentType]
    else:
        annotation_text_iter = annotation_text

    # Get coordinates for secondary structure plot
    coordinates = RNAplotApp.compute_coordinates(
        base_pairs=unknotted_base_pairs,
        length=length,
        bin_path=bin_path,
        layout_type=layout_type,
    )

    # Rotate Coordinates
    if angle != 0:
        angle = np.deg2rad(angle)
        rot_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        for i, coord in enumerate(coordinates):
            coordinates[i] = np.dot(rot_matrix, coord)

    # Remove axes and frame
    axes.set_xticks([])
    axes.set_yticks([])
    axes.xaxis.set_tick_params(
        top=False, bottom=False, labeltop=False, labelbottom=False
    )
    axes.yaxis.set_tick_params(
        top=False, bottom=False, labeltop=False, labelbottom=False
    )
    axes.set_frame_on(False)

    # Define buffer area (Border)
    coord_range = abs(np.max(coordinates)) + abs(np.min(coordinates))
    buffer = border * coord_range

    # Adjust display
    axes.set_xlim(
        float(np.min(coordinates[:, 0]) - buffer),
        float(np.max(coordinates[:, 0]) + buffer),
    )
    axes.set_ylim(
        float(np.min(coordinates[:, 1]) - buffer),
        float(np.max(coordinates[:, 1]) + buffer),
    )
    axes.set_aspect(aspect="equal")

    # Draw backbone
    axes.plot(
        coordinates[:, 0],
        coordinates[:, 1],
        color=backbone_color,
        linestyle=backbone_linestyle,
        linewidth=backbone_linewidth,
    )

    # Draw base labels
    for coords, label, box, text in zip(
        coordinates, base_labels, base_box_iter, base_text_iter
    ):
        t = axes.text(
            x=coords[0], y=coords[1], s=label, ha="center", va="center", **text
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
        axes.plot(x, y, color=color, linestyle=style, linewidth=width)

    # Draw annotations
    for i, text in zip(annotation_positions, annotation_text_iter):
        if (i > 0) and ((i + 1) < length):
            # Get the average of the direction vectors to the next and
            # previous base
            vector_to_previous = np.array(
                [
                    coordinates[i - 1][0] - coordinates[i][0],
                    coordinates[i - 1][1] - coordinates[i][1],
                ]
            )
            vector_to_previous = vector_to_previous / np.linalg.norm(vector_to_previous)
            vector_to_next = np.array(
                [
                    coordinates[i][0] - coordinates[i + 1][0],
                    coordinates[i][1] - coordinates[i + 1][1],
                ]
            )
            vector_to_next = vector_to_next / np.linalg.norm(vector_to_next)
            vector = (vector_to_next + vector_to_previous) / 2
        elif i > 0:
            # For the last base get the direction vector to the previous
            # base
            vector = np.array(
                [
                    coordinates[i - 1][0] - coordinates[i][0],
                    coordinates[i - 1][1] - coordinates[i][1],
                ]
            )
        else:
            # For the first base get the direction vector to the next
            # base
            vector = np.array(
                [
                    coordinates[i][0] - coordinates[i + 1][0],
                    coordinates[i][1] - coordinates[i + 1][1],
                ]
            )
        # Normalize the vector
        vector = vector / np.linalg.norm(vector)
        # Get the perpendicular vector
        vector = np.array([vector[1], -vector[0]])
        # The annotations are offset in the direction of the
        # perpendicular vector
        x, y = coordinates[i] + (annotation_offset * vector)
        axes.text(x=x, y=y, s=str(i + 1), ha="center", va="center", **text)
