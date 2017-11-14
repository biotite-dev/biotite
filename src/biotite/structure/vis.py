# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

"""
Contains functions for visualisation of structures and structure data.
"""

__all__ = ["simple_view"]


def simple_view(fig, atom_arrays):
    """
    Show the structure using a `matplotlib` 3-D plot.
    
    This is a very simply visualisation function.
    
    Parameters
    ----------
    fig : Figure
        The `Figure` to plot the structures in.
    atom_arrays : Iterable object of AtomArray, e.g. AtomArrayStack
        Structures to plot.
    
    Returns
    -------
    ax : Axes
        Axes containing the visualised structures.
    """
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.gca(projection='3d')
    ax.axis("off")
    ax.set_aspect("equal")
    # Sets equal limits to all dimension, since set_aspect("equal") does not work properly
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-10,10)
    for atom_array in atom_arrays:
        atom_array = atom_array[(atom_array.atom_name == "CA")]
        ax.plot(atom_array.coord[:,0],
                 atom_array.coord[:,1],
                 atom_array.coord[:,2],)
    return ax