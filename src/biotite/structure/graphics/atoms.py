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


def plot_atoms(axes, atoms, colors, line_width=1.0):
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
    
    Notes
    -----
    This is a very simple visualization tools for quick visual analysis
    of a structure.
    For publication-ready molecular images the usage of a dedicated
    molecular visualization tool is recommended.
    """
    if atoms.bonds is None:
        raise ValueError("The atom array must have an associated bond list")
    

    line_coord = []
    line_colors = []
    for index1, index2 in atoms.bonds.as_array()[:,:2]:
        line_start = atoms.coord[index1]
        line_end = atoms.coord[index2]
        line_center = (line_start + line_end) / 2
        
        line_coord.append((
        line_start, line_center
        ))
        line_colors.append(colors[index1])
        
        line_coord.append((
        line_end, line_center
        ))
        line_colors.append(colors[index2])


    lines = Line3DCollection(
        line_coord, color=line_colors, linewidths=line_width
    )
    axes.add_collection(lines)
    axes.axis("off")
    _set_equal_aspect(axes, atoms.coord)


def _set_equal_aspect(axes, coord):
    """
    Fix to ensure equal aspect ratio in a 3D plot under the condition,
    that the :class:`Axes` is quadratic on the display.
    """
    max_range = np.array(
        [coord[:, 0].max() - coord[:, 0].min(),
         coord[:, 1].max() - coord[:, 1].min(),
         coord[:, 2].max() - coord[:, 2].min()]
    ).max() / 2

    center_x = (coord[:, 0].max() + coord[:, 0].min()) / 2
    center_y = (coord[:, 1].max() + coord[:, 1].min()) / 2
    center_z = (coord[:, 2].max() + coord[:, 2].min()) / 2
    axes.set_xlim(center_x - max_range, center_x + max_range)
    axes.set_ylim(center_y - max_range, center_y + max_range)
    axes.set_zlim(center_z - max_range, center_z + max_range)