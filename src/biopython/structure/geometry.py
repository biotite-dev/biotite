# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This module provides functions for geometric measurements between atoms in a
structure, mainly lenghts and angles.
"""

import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack, coord
from .util import vector_dot, norm_vector
from .filter import filter_backbone
from .error import BadStructureError

def distance(atoms1, atoms2):
    """
    Measures the euclidian distance between atoms.
    
    Parameters
    ----------
    atoms1, atoms2 : ndarray or Atom or AtomArray or AtomArrayStack
        The atoms to measure the distances between. The dimensions may vary.
        Alternatively an ndarray containing the coordinates can be provided.
    
    Returns
    -------
    dist : float or ndarray
        The atom distances. The shape is equal to the shape of
        the input `atoms` with the highest dimensionality minus the last axis.
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
    Measures the centroid of a structure.
    
    Parameters
    ----------
    atoms: ndarray orAtomArray or AtomArrayStack
        The structures to determine the centroid from.
        Alternatively an ndarray containing the coordinates can be provided.
    
    Returns
    -------
    centroid : float or ndarray
        The centroid of the structure(s). The shape is equal to the shape of
        the input `atoms` with the highest dimensionality minus the second last axis.
    """
    return np.mean(coord(atoms), axis=-2)


def angle(atom1, atom2, atom3):
    """
    Measures the angle between 3 atoms.
    
    Parameters
    ----------
    atoms1, atoms2, atoms3 : ndarray or Atom or AtomArray or AtomArrayStack
        The atoms to measure the angle between.
        Alternatively an ndarray containing the coordinates can be provided.
    
    Returns
    -------
    angle : float or ndarray
        The angle(s) between the atoms. The shape is equal to the shape of
        the input `atoms` with the highest dimensionality minus the last axis.
    """
    v1 = coord(atom2) - coord(atom1)
    v2 = coord(atom3) - coord(atom2)
    norm_vector(v1)
    norm_vector(v2)
    return np.arccos(vector_dot(v1,v2))


def dihedral(atom1, atom2, atom3, atom4):
    """
    Measures the dihedral angle between 4 atoms.
    
    Parameters
    ----------
    atoms1, atoms2, atoms3, atom4 : ndarray or Atom or AtomArray or AtomArrayStack
        The atoms to measure the dihedral angle between.
        Alternatively an ndarray containing the coordinates can be provided.
    
    Returns
    -------
    dihed : float or ndarray
        The dihedral angle(s) between the atoms. The shape is equal to the shape of
        the input `atoms` with the highest dimensionality minus the last axis.
    
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
    
    x = vector_dot(n1,n2)
    y = vector_dot(np.cross(n1,n2), v2)
    
    return np.arctan2(y,x)


def dihedral_backbone(atom_array, chain_id):
    """
    Measures the characteristic backbone dihedral angles of a structure.
    
    Parameters
    ----------
    atom_array: AtomArray
        The protein structure.
    chain_id: string
        The ID of the polypeptide chain. The dihedral angles are calculated
        for ``atom_array[atom_array.chain_id == chain_id]``
    
    Returns
    -------
    psi : 1-D ndarray
        An array containing the *psi* dihedral angles. The index *n* represents
        the section between CA(n+1) and CA(n+2). Therefore the length of this
        array is the length of the AtomArray -1.
    omega : 1-D ndarray
        An array containing the *omega* dihedral angles. Indexing as above.
    phi : 1-D ndarray
        An array containing the *phi* dihedral angles. Indexing as above.
    
    Raises
    ------
    BadStructureError
        If the amount of backbone atoms is not equal to amount of residues*3
        (for N, CA and C).
    
    See Also
    --------
    dihedral
    """
    try:
        # Filter all backbone atoms
        backbone = atom_array[filter_backbone(atom_array) &
                              (atom_array.chain_id == chain_id)]
        angle_atoms = np.zeros(( len(backbone)-3, 4, 3 ))
        # Fill numpy array, where last dimension is used for dihedral calc
        for i in range(len(angle_atoms)):
            angle_atoms[i] = backbone.coord[0+i : 4+i]
        dihed = dihedral(angle_atoms[:,0], angle_atoms[:,1],
                         angle_atoms[:,2], angle_atoms[:,3])
        # Extract the three dihedral angle types
        psi = dihed[0::3]
        omega = dihed[1::3]
        phi = dihed[2::3]
        return psi, omega, phi
    except Exception as err:
        if len(backbone) != backbone.seq_length()*3:
            raise BadStructureError("AtomArray has insufficient amount"
                "of backbone atoms") from None
        else:
            raise