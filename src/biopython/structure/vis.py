# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
Contains functions for visualisation of structures and structure data.
"""

def simple_view(fig, atom_arrays):
    """
    Show the structure using a `matplotlib` 3-D plot.
    
    This is a very simply visualisation function, which is best suited for
    AtomArrays containing only the backbone atoms.
    
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
        atom_array = atom_array[(atom_array.atom_name == " CA ")]
        ax.plot(atom_array.coord[:,0],
                 atom_array.coord[:,1],
                 atom_array.coord[:,2],)
    return ax