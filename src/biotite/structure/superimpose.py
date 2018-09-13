# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for structure superimposition.
"""

__author__ = "Patrick Kunzmann"
__all__ = ["superimpose", "superimpose_apply"]

import numpy as np
from .geometry import centroid
from .atoms import Atom, AtomArray, AtomArrayStack, stack
from .error import BadStructureError


def superimpose(fixed, mobile, atom_mask=None):
    """
    Superimpose structures onto a fixed structure.
    
    The superimposition is performed using the Kabsch algorithm
    [1]_ [2]_, so that the RMSD between the superimposed and the fixed
    structure is minimized.
    
    Parameters
    ----------
    fixed : AtomArray
        The fixed structure.
    mobile: AtomArray or AtomArrayStack
        The structure(s) which is/are superimposed on the `fixed`
        structure. Both, `fixed` and `mobile` should have equal
        annotation arrays and must have equal sizes.
    atom_mask: ndarray, dtype=bool, optional
        If given, only the atoms covered by this boolean mask will be
        considered for superimposition.
        This means that the algorithm will minimize the RMSD based
        on the covered atoms instead of all atoms.
        The returned superimposed structure will contain all atoms
        of the input structure, regardless of this parameter.
    
    Returns
    -------
    fitted : AtomArray or AtomArrayStack
        A copy of the `mobile` structure(s),
        superimposed on the fixed structure.
    transformation : tuple or tuple list
        The tuple contains the transformations that were applied on
        `mobile`. This can be used in `apply_superimposition()` in order
        to transform another AtomArray in the same way.
        The first element contains the translation vector for moving the
        centroid into the origin.
        The second element contains the rotation matrix.
        The third element contains the translation vector for moving the
        structure onto the fixed.
        The three transformations are performed sequentially.
    
    See Also
    --------
    superimpose_apply
    
    Notes
    -----
    The `transformation` tuple can be used in `superimpose_apply()` in
    order to transform another `AtomArray` in the same way. This can
    come inhandy, in case you want to superimpose two structures with
    different amount of atoms. Often the two structures can be sliced in
    order to obtain the same size and annotation arrays. After
    superimposition the transformation can be applied on the original
    structure using `superimpose_apply()`.
    
    References
    ----------
    
    .. [1] W Kabsch,
       "A solution for the best rotation to relate two sets of vectors."
       Acta Cryst, 32, 922-923 (1976).
       
    .. [2] W Kabsch,
       "A discussion of the solution for the best rotation to relate
       two sets of vectors."
       Acta Cryst, 34, 827-828 (1978).
    
    Examples
    --------
    
    Superimpose two models of 20 residue peptide, randomly
    rotate/translate one of them and superimpose it onto the other
    model:
    
    >>> file = PDBxFile()
    >>> file.read("1l2y.cif")
    >>> array1 = get_structure(file, model=1)
    >>> array2 = get_structure(file, model=2)
    >>> array2 = translate(array2, [1,2,3])
    >>> array2 = rotate(array2, [1,2,3])
    >>> print(rmsd(array1, array2))
    10.8492095649
    >>> array2_fit, transformation = superimpose(
            array1, array2, mask=(array2.atom_name == "CA")
        )
    >>> print(rmsd(array1, array2_fit))
    1.95480879468
    >>> array2_fit, transformation = superimpose(array1, array2)
    >>> print(rmsd(array1, array2_fit))
    1.92792691375
        
    """
    mob_centroid = centroid(mobile)
    fix_centroid = centroid(fixed)
    if atom_mask is not None:
        # Implicitly this creates array copies
        mob_centered = mobile.coord[..., atom_mask, :]
        fix_centered = fixed.coord[..., atom_mask, :]
    else:
        mob_centered = np.copy(mobile.coord)
        fix_centered = np.copy(fixed.coord)

    if mob_centered.shape[-2] != fix_centered.shape[-2]:
        raise BadStructureError(
            f"The mobile array ({mob_centered.shape[-2]} atoms) "
            f"and the fixed array ({fix_centered.shape[-2]} atoms), "
            f"have an unequal amount of atoms"
        )
    
    mob_centered -= mob_centroid[..., np.newaxis, :]
    fix_centered -= fix_centroid
    
    if not isinstance(fixed, AtomArray):
        raise ValueError("Reference must be AtomArray")
   
    if isinstance(mobile, AtomArray):
        # Simply superimpose without loop
        rotation = _superimpose(fix_centered, mob_centered)
        superimposed = mobile.copy()
        superimposed.coord -= mob_centroid[..., np.newaxis, :]
        superimposed.coord = np.dot(superimposed.coord, rotation)
        superimposed.coord += fix_centroid
        return superimposed, (-mob_centroid, rotation, fix_centroid)
    
    elif isinstance(mobile, AtomArrayStack):
        superimposed = mobile.copy()
        superimposed.coord -= mob_centroid[..., np.newaxis, :]
        # Perform Kabsch algorithm for every model
        transformations = [None] * len(superimposed.coord)
        for i in range(len(superimposed.coord)):
            rotation = _superimpose(fix_centered, mob_centered[i])
            superimposed.coord[i] = np.dot(superimposed.coord[i], rotation)
            transformations[i] = (-mob_centroid[i], rotation, fix_centroid)
        superimposed.coord += fix_centroid
        return superimposed, transformations
   
    else:
        raise ValueError("Mobile structure must be AtomArray "
                         "or AtomArrayStack")


def _superimpose(fix_centered, mob_centered):
    """
    Perform the Kabsch algorithm using only the coordinates.
    """
    # Calculating rotation matrix
    y = mob_centered
    x = fix_centered
    # Calculate covariance matrix
    cov = np.dot(y.T, x)
    v, s, w = np.linalg.svd(cov)
    # Remove possibility of reflected atom coordinates
    if np.linalg.det(v) * np.linalg.det(w) < 0:
        v[:,-1] *= -1
    rotation = np.dot(v,w)
    return rotation


def superimpose_apply(atoms, transformation):
    """
    Superimpose structures using a given transformation tuple.
    
    The transformation tuple is obtained by prior superimposition.
    
    Parameters
    ----------
    atoms : AtomArray
        The structure to apply the transformation on.
    transformation: tuple, size=3
        The transfomration tuple, obtained by `superimpose()`.
    
    Returns
    -------
    fitted : AtomArray or AtomArrayStack
        A copy of the `atoms` structure,
        with transformations applied.
    
    See Also
    --------
    superimpose
    """
    transformed = atoms.copy()
    transformed.coord += transformation[0]
    transformed.coord = np.dot(transformed.coord, transformation[1])
    transformed.coord += transformation[2]
    return transformed
