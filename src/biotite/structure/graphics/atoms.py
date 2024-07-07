# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.graphics"
__author__ = "Patrick Kunzmann"
__all__ = ["plot_atoms", "plot_ball_and_stick_model"]

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_atoms(
    axes,
    atoms,
    colors,
    line_width=1.0,
    background_color=None,
    center=None,
    size=None,
    zoom=1.0,
):
    """
    Plot an :class:`AtomArray` as lines between bonded atoms.

    The z-axis points into the screen plane.

    Parameters
    ----------
    axes : Axes3D
        The *Matplotlib* 3D-axes to plot the structure on.
    atoms : AtomArray, shape=(n,)
        The structure to be plotted.
        The atom array must have an associated :class:`BondList`, i.e.
        its ``bonds`` attribute must be defined.
        One line for each bond is drawn.
    colors : ndarray, shape=(n,3) or shape=(n,4), dtype=float
        An array of RGB or RGBA colors for each atom in `atoms`.
        The values for each color channel are in the range 0 to 1.
    line_width : float, optional
        The width of the lines to be drawn.
    background_color : object
        A *Matplotlib* compatible color (color name or RGB values).
        If set, the background is colored with the given value.
    center : tuple of (float, float, float), optional
        The coordinates of the structure that are in the center of the
        plot.
        By default the complete molecule is centered.
    size : float, optional
        The size of each dimension in the plot.
        The limits of the :class:`Axes3D` are set to
        ``(center - size/(2*zoom)), (center + size/(2*zoom))``.
    zoom : float, optional
        Zoom in or zoom out.

        - ``> 1.0``: Zoom in.
        - ``< 1.0``: Zoom out.

    Notes
    -----
    This is a very simple visualization tools for quick visual analysis
    of a structure.
    For publication-ready molecular images the usage of a dedicated
    molecular visualization tool is recommended.
    """
    if not isinstance(axes, Axes3D):
        raise ValueError("The given axes mut be an 'Axes3D'")
    if atoms.bonds is None:
        raise ValueError("The atom array must have an associated bond list")

    # Calculating connections between atoms
    line_coord = []
    line_colors = []
    for index1, index2 in atoms.bonds.as_array()[:, :2]:
        # Every connection consist of two lines:
        # One from the first atom to the center
        # and from from the second atom to the center
        line_start = atoms.coord[index1]
        line_end = atoms.coord[index2]
        line_center = (line_start + line_end) / 2

        # Add line from first atom
        line_coord.append((line_start, line_center))
        line_colors.append(colors[index1])

        # Add line from second atom
        line_coord.append((line_end, line_center))
        line_colors.append(colors[index2])

    # Plot computed line coordinates and colors
    # Use 'Line3DCollection' for higher efficiency
    lines = Line3DCollection(
        line_coord,
        color=line_colors,
        linewidths=line_width,
        capstyle="round",
        joinstyle="round",
    )
    axes.add_collection(lines)

    # Set viewing angle
    axes.azim = -90
    axes.elev = 90
    # Remove frame
    axes.axis("off")
    # Set background color
    if background_color is not None:
        axes.set_facecolor(background_color)
        axes.get_figure().set_facecolor(background_color)
    _set_box(axes, atoms.coord, center, size, zoom)


def plot_ball_and_stick_model(
    axes,
    atoms,
    colors,
    ball_size=200,
    line_color="black",
    line_width=1.0,
    background_color=None,
    center=None,
    size=None,
    zoom=1.0,
):
    """
    Plot an :class:`AtomArray` as *ball-and-stick* model.

    The z-axis points into the screen plane.

    UNSTABLE: This function is probably subject to future changes.

    Parameters
    ----------
    axes : Axes3D
        The *Matplotlib* 3D-axes to plot the structure on.
    atoms : AtomArray, length=n
        The structure to be plotted.
        The atom array must have an associated :class:`BondList`, i.e.
        its ``bonds`` attribute must be defined.
        One line for each bond is drawn.
    colors : ndarray, shape=(n,3) or shape=(n,4), dtype=float
        An array of RGB or RGBA colors for each atom in `atoms`, that
        is used to color the *balls* of the model.
        The values for each color channel are in the range 0 to 1.
    ball_size : int or iterable of int, shape=(n,)
        The size of the *balls* in the model in *pt*.
        Either a single value for all atoms or an iterable object of
        values for each atom.
    line_color : object
        A *Matplotlib* compatible color value for the *sticks* of the
        model.
    line_width : float, optional
        The width of the *sticks* in the model in *pt*.
    background_color : object
        A *Matplotlib* compatible color (color name or RGB values).
        If set, the background is colored with the given value.
    center : tuple of (float, float, float), optional
        The coordinates of the structure that are in the center of the
        plot.
        By default the complete molecule is centered.
    size : float, optional
        The size of each dimension in the plot.
        The limits of the :class:`Axes3D` are set to
        ``(center - size/(2*zoom)), (center + size/(2*zoom))``.
    zoom : float, optional
        Zoom in or zoom out.

        - ``> 1.0``: Zoom in.
        - ``< 1.0``: Zoom out.

    Notes
    -----
    This is a very simple visualization tools for quick visual analysis
    of a structure.
    For publication-ready molecular images the usage of a dedicated
    molecular visualization tool is recommended.
    """
    if not isinstance(axes, Axes3D):
        raise ValueError("The given axes mut be an 'Axes3D'")
    if atoms.bonds is None:
        raise ValueError("The atom array must have an associated bond list")

    # Calculating connections between atoms
    line_coord = [
        (atoms.coord[index1], atoms.coord[index2])
        for index1, index2 in atoms.bonds.as_array()[:, :2]
    ]

    # Plot sticks
    # Use 'Line3DCollection' for higher efficiency
    sticks = Line3DCollection(
        line_coord,
        color=line_color,
        linewidths=line_width,
        capstyle="round",
        joinstyle="round",
    )
    axes.add_collection(sticks)

    # Plot balls
    axes.scatter(*atoms.coord.T, s=ball_size, c=colors, linewidth=0, alpha=1)

    # Set viewing angle
    axes.azim = -90
    axes.elev = 90
    # Remove frame
    axes.axis("off")
    # Set background color
    if background_color is not None:
        axes.set_facecolor(background_color)
        axes.get_figure().set_facecolor(background_color)
    _set_box(axes, atoms.coord, center, size, zoom)


def _set_box(axes, coord, center, size, zoom):
    """
    This ensures an approximately equal aspect ratio in a 3D plot under
    the condition, that the :class:`Axes` is quadratic on the display.
    """
    if center is None:
        center = (
            (coord[:, 0].max() + coord[:, 0].min()) / 2,
            (coord[:, 1].max() + coord[:, 1].min()) / 2,
            (coord[:, 2].max() + coord[:, 2].min()) / 2,
        )

    if size is None:
        size = np.array(
            [
                coord[:, 0].max() - coord[:, 0].min(),
                coord[:, 1].max() - coord[:, 1].min(),
                coord[:, 2].max() - coord[:, 2].min(),
            ]
        ).max()

    axes.set_xlim(center[0] - size / (2 * zoom), center[0] + size / (2 * zoom))
    axes.set_ylim(center[1] - size / (2 * zoom), center[1] + size / (2 * zoom))
    axes.set_zlim(center[2] - size / (2 * zoom), center[2] + size / (2 * zoom))

    # Make the axis lengths of the 'plot box' equal
    # The 'plot box' is not visible due to 'axes.axis("off")'
    axes.set_box_aspect([1, 1, 1])
