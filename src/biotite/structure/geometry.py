# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for geometric measurements between atoms
in a structure, mainly lenghts and angles.
"""

__author__ = "Patrick Kunzmann"
__all__ = ["distance", "centroid", "angle", "dihedral", "dihedral_backbone"]

import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack, coord
from .util import vector_dot, norm_vector
from .filter import filter_backbone
from .error import BadStructureError


def distance(atoms1, atoms2):
    """
    Measure the euclidian distance between atoms.
    
    Parameters
    ----------
    atoms1, atoms2 : ndarray or Atom or AtomArray or AtomArrayStack
        The atoms to measure the distances between. The dimensions may
        vary. Alternatively an ndarray containing the coordinates can be
        provided.
    
    Returns
    -------
    dist : float or ndarray
        The atom distances. The shape is equal to the shape of
        the input `atoms` with the highest dimensionality minus the last
        axis.
    """
    v1 = coord(atoms1)
    v2 = coord(atoms2)
    # decide subtraction order based on shape,
    # since an array can be only subtracted by an array with less dimensions
    if len(v1.shape) <= len(v2.shape):
        dif = v2 - v1
    else:
        dif = v1 - v2
    dist = np.sqrt(vector_dot(dif, dif))
    return dist


def centroid(atoms):
    """
    Measure the centroid of a structure.
    
    Parameters
    ----------
    atoms: ndarray or AtomArray or AtomArrayStack
        The structures to determine the centroid from.
        Alternatively an ndarray containing the coordinates can be
        provided.
    
    Returns
    -------
    centroid : float or ndarray
        The centroid of the structure(s). `ndarray` is returned when
        an `AtomArrayStack` is given (centroid for each model).
    """
    return np.mean(coord(atoms), axis=-2)


def angle(atom1, atom2, atom3):
    """
    Measure the angle between 3 atoms.
    
    Parameters
    ----------
    atoms1, atoms2, atoms3 : ndarray or Atom or AtomArray or AtomArrayStack
        The atoms to measure the angle between. Alternatively an
        ndarray containing the coordinates can be provided.
    
    Returns
    -------
    angle : float or ndarray
        The angle(s) between the atoms. The shape is equal to the shape
        of the input `atoms` with the highest dimensionality minus the
        last axis.
    """
    v1 = coord(atom1) - coord(atom2)
    v2 = coord(atom3) - coord(atom2)
    norm_vector(v1)
    norm_vector(v2)
    return np.arccos(vector_dot(v1,v2))


def dihedral(atom1, atom2, atom3, atom4):
    """
    Measure the dihedral angle between 4 atoms.
    
    Parameters
    ----------
    atoms1, atoms2, atoms3, atom4 : ndarray or Atom or AtomArray or AtomArrayStack
        The atoms to measure the dihedral angle between.
        Alternatively an ndarray containing the coordinates can be
        provided.
    
    Returns
    -------
    dihed : float or ndarray
        The dihedral angle(s) between the atoms. The shape is equal to
        the shape of the input `atoms` with the highest dimensionality
        minus the last axis.
    
    See Also
    --------
    dihedral_backbone
    """
    v1 = coord(atom2) - coord(atom1)
    v2 = coord(atom3) - coord(atom2)
    v3 = coord(atom4) - coord(atom3)
    norm_vector(v1)
    norm_vector(v2)
    norm_vector(v3)
    
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    
    # Calculation using atan2, to ensure the correct sign of the angle 
    x = vector_dot(n1,n2)
    y = vector_dot(np.cross(n1,n2), v2)
    return np.arctan2(y,x)


def dihedral_backbone(atom_array, chain_id):
    """
    Measure the characteristic backbone dihedral angles of a structure.
    
    Parameters
    ----------
    atom_array: AtomArray or AtomArrayStack
        The protein structure. A complete backbone, without gaps,
        is required here.
    chain_id: string
        The ID of the polypeptide chain. The dihedral angles are
        calculated for ``atom_array[atom_array.chain_id == chain_id]``
    
    Returns
    -------
    phi, psi, omega : ndarray
        An array containing the 3 backbone dihedral angles for every
        CA. 'phi' is not defined at the N-terminus, 'psi' and 'omega'
        are not defined at the C-terminus. In these places the arrays
        have `NaN` values. If an `AtomArrayStack` is given, the output
        angles are 2-dimensional, the first dimension corresponds to
        the model number.
    
    Raises
    ------
    BadStructureError
        If the amount of backbone atoms is not equal to amount of
        residues times 3 (for N, CA and C).
    
    See Also
    --------
    dihedral
    
    Examples
    --------
    
    Input
    
    >>> pdbx_file = PDBxFile()
    >>> pdbx_file.read("1l2y.cif")
    >>> atom_array = get_structure(pdbx_file, model=1)
    >>> psi, omega, phi = struc.dihedral_backbone(atom_array, "A")
    >>> plt.plot(phi * 360/(2*np.pi), psi * 360/(2*np.pi),
    ...          linestyle="None", marker="o")
    >>> plt.xlim(-180, 180)
    >>> plt.ylim(-180, 180)
    >>> plt.xlabel("phi")
    >>> plt.ylabel("psi")
    >>> plt.show()
    
    Output
    
    .. image:: /static/assets/figures/dihedral.svg
    """
    # Filter all backbone atoms
    bb_coord = atom_array[...,
                            filter_backbone(atom_array) &
                            (atom_array.chain_id == chain_id)].coord
    if bb_coord.shape[-1] % 3 != 0:
        raise BadStructureError(
            "AtomArray has insufficient amount of backbone atoms "
            "(possibly missing terminus)"
        )
    
    # Coordinates for dihedral angle calculation
    # Dim 0: Model index (only for atom array stacks)
    # Dim 1: Angle index
    # Dim 2: X, Y, Z coordinates
    # Dim 3: Atoms involved in dihedral angle
    if isinstance(atom_array, AtomArray):
        angle_coord_shape = (len(bb_coord)//3, 3, 4)
    elif isinstance(atom_array, AtomArrayStack):
        angle_coord_shape = (bb_coord.shape[0], bb_coord.shape[1]//3, 3, 4)
    phi_coord   = np.full(angle_coord_shape, np.nan)
    psi_coord   = np.full(angle_coord_shape, np.nan)
    omega_coord = np.full(angle_coord_shape, np.nan)
    
    # Indices for coordinates of CA atoms 
    ca_i = np.arange(bb_coord.shape[-2]//3) * 3 + 1
    phi_coord  [..., 1: , :, 0]  = bb_coord[..., ca_i[1: ]-2 ,:]
    phi_coord  [..., 1: , :, 1]  = bb_coord[..., ca_i[1: ]-1 ,:]
    phi_coord  [..., 1: , :, 2]  = bb_coord[..., ca_i[1: ]   ,:]
    phi_coord  [..., 1: , :, 3]  = bb_coord[..., ca_i[1: ]+1 ,:]
    psi_coord  [..., :-1, :, 0]  = bb_coord[..., ca_i[:-1]-1 ,:]
    psi_coord  [..., :-1, :, 1]  = bb_coord[..., ca_i[:-1]   ,:]
    psi_coord  [..., :-1, :, 2]  = bb_coord[..., ca_i[:-1]+1 ,:]
    psi_coord  [..., :-1, :, 3]  = bb_coord[..., ca_i[:-1]+2 ,:]
    omega_coord[..., :-1, :, 0]  = bb_coord[..., ca_i[:-1]   ,:]
    omega_coord[..., :-1, :, 1]  = bb_coord[..., ca_i[:-1]+1 ,:]
    omega_coord[..., :-1, :, 2]  = bb_coord[..., ca_i[:-1]+2 ,:]
    omega_coord[..., :-1, :, 3]  = bb_coord[..., ca_i[:-1]+3 ,:]
    
    phi = dihedral(phi_coord[...,0], phi_coord[...,1],
                    phi_coord[...,2], phi_coord[...,3])
    psi = dihedral(psi_coord[...,0], psi_coord[...,1],
                    psi_coord[...,2], psi_coord[...,3])
    omega = dihedral(omega_coord[...,0], omega_coord[...,1],
                        omega_coord[...,2], omega_coord[...,3])
    
    return phi, psi, omega
