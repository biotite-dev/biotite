# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

"""
This module provides functions for structure superimposition.
"""

import numpy as np
from .geometry import centroid
from .atoms import Atom, AtomArray, AtomArrayStack, stack

__all__ = ["superimpose", "superimpose_apply"]


def superimpose(fixed, mobile, ca_only=True):
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
    ca_only: bool, optional
        If True, the function performs the superimposition
        considering only the "CA" atoms. This increases the
        performance drastically but decreases the accuracy slightly.
        True by default.
    
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
        >>> array2_fit, transformation = superimpose(array1, array2, ca_only=True)
        >>> print(rmsd(array1, array2_fit))
        1.95480879468
        >>> array2_fit, transformation = superimpose(array1, array2, ca_only=False)
        >>> print(rmsd(array1, array2_fit))
        1.92792691375
        
    """
    if type(fixed) != AtomArray:
        raise ValueError("Reference must be AtomArray")
    if type(mobile) == AtomArray:
        # Simply superimpose without loop
        return _superimpose(fixed, mobile, ca_only)
    elif type(mobile) == AtomArrayStack:
        # Superimpose for every mobile in AtomArrayStack
        fitted_subjects = []
        transformations = []
        for mobile in mobile:
            fitted_subject, transformation = _superimpose(fixed,
                                                          mobile, ca_only)
            fitted_subjects.append(fitted_subject)
            transformations.append(transformation)
        # Convert AtomArray list back to AtomArrayStack
        fitted_subjects = stack(fitted_subjects)
        return (fitted_subjects, transformations)
    else:
        raise ValueError("Reference must be AtomArray")


def _superimpose(fixed, mobile, ca_only):
    """
    Performs the actual superimposition.
    """
    if type(mobile) == AtomArray:
        mob_centroid = centroid(mobile)
        fix_centroid = centroid(fixed)
        if ca_only:
            # For performance reasons the Kabsch algorithm
            # is only performed with "CA" per default
            # Implicitly this creates array copies
            mob_centered = mobile.coord[(mobile.atom_name == "CA")]
            fix_centered = fixed.coord[(fixed.atom_name == "CA")]
        else:
            mob_centered = np.copy(mobile.coord)
            fix_centered = np.copy(fixed.coord)
            
        if len(mob_centered) != len(fix_centered):
            raise BadStructureException("The mobile and fixed array "
                                        "have different amount of atoms")
        
        mob_centered -= mob_centroid
        fix_centered -= fix_centroid
        
        # Calculating rotation matrix using Kabsch algorithm
        y = mob_centered
        x = fix_centered
        # Calculate covariance matrix
        cov = np.dot(y.T, x)
        v, s, w = np.linalg.svd(cov)
        rotation = np.dot(w, v.T)
        # Remove possibility of reflected atom coordinates
        if np.linalg.det(v) * np.linalg.det(w) < 0:
            s[-1,:] *= -1
            v[:,-1] *= -1
        rotation = np.dot(v,w)
        
        fitted_subject = mobile.copy()
        fitted_subject.coord -= mob_centroid
        fitted_subject.coord = np.dot(fitted_subject.coord, rotation)
        fitted_subject.coord += fix_centroid
        return fitted_subject, (-mob_centroid, rotation, fix_centroid)


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
