# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.graphics"
__author__ = "Patrick Kunzmann"
__all__ = ["plot_atoms"]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_atoms(axes, atoms, colors, line_width=1.0, background_color=None,
               center=None, size=None):
    """
    Plot an :class:`AtomArray` as lines between bonded atoms.
    
    Parameters
    ----------
    axes : Axes3D
        The *Matplotlib* 3D-axes to plot the structure on.
        For a correct aspect ratio of the visualized structure, the
        :class:`Axes3D` must have quadratic extents on the display.
        This can be approximately achieved by giving the :class:`Figure`
        equal values for the width and height in its `figsize`.
    atoms : AtomArray, length=n
        The structure to be plotted.
        The atom array must have an associated :class:`BondList`, i.e.
        its `bonds` attribute must be defined.
        One line for each bond is drawn.
    colors : ndarray, shape=(n,3) or shape=(n,4), dtype=float
        An array of RGB or RGBA colors for each atom in `atoms`.
        The values for each color channel are in the range 0 to 1.
    line_width : float, optional
        The width of the lines to be drawn.
    background_color : string or iterable object
        A matplotlib compatible color (color name or RGB values).
        If set, the background is colored with the given value.
    center : tuple of (float, float, float), optional
        The coordinates of the structure that are in the center of the
        plot.
        By default the complete molecule is centered.
    size : float
        The size of each dimension in the plot.
        The limits of the :class:`Axes3D` are set to
        ``(center - size/2), (center + size/2)``.
    
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
    for index1, index2 in atoms.bonds.as_array()[:,:2]:
        # Every connection consist of two lines:
        # One from the first atom to the center
        # and from from the second atom to the center
        line_start = atoms.coord[index1]
        line_end = atoms.coord[index2]
        line_center = (line_start + line_end) / 2
        
        # Add line from first atom
        line_coord.append((
            line_start, line_center
        ))
        line_colors.append(colors[index1])
        
        # Add line from second atom
        line_coord.append((
            line_end, line_center
        ))
        line_colors.append(colors[index2])

    # Plot computed line coordinates and colors
    # Use 'Line3DCollection' for higher efficiency
    lines = Line3DCollection(
        line_coord, color=line_colors, linewidths=line_width
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
    _set_box(axes, atoms.coord, center, size)


def _set_box(axes, coord, center, size):
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
        size = np.array([
            coord[:, 0].max() - coord[:, 0].min(),
            coord[:, 1].max() - coord[:, 1].min(),
            coord[:, 2].max() - coord[:, 2].min()
        ]).max()

    axes.set_xlim(center[0] - size/2, center[0] + size/2)
    axes.set_ylim(center[1] - size/2, center[1] + size/2)
    axes.set_zlim(center[2] - size/2, center[2] + size/2)