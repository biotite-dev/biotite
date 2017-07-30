# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This module provides functions for structure superimposition.
"""

import numpy as np
from .geometry import centroid
from .atoms import Atom, AtomArray, AtomArrayStack, stack

def superimpose(reference, subject, ca_only=True):
    """
    Superimpose structures on a reference structure.
    
    The superimposition is performed using the Kabsch algorithm [1]_ [2]_.
    
    Parameters
    ----------
    reference : AtomArray
        The reference structure.
    subject: AtomArray or AtomArrayStack
        The structure(s) which is/are superimposed on the `reference`
        structure. Both, `reference` and `subject` should have equal
        annotation arrays and must have equal sizes.
    ca_only: bool, optional
        If True, the function performs the superimpostion
        considering only the "CA" atoms. This increases the
        performance drastically but decreases the accuracy slightly.
    
    Returns
    -------
    fitted : AtomArray or AtomArrayStack
        A copy of the `subject` structure(s),
        superimposed on the reference structure.
    transformation : tuple or tuple list
        The tuple contains the transformations that were applied on `subject`.
        This can be used in `apply_superimposition()` in order to transform
        another AtomArray in the same way.
        The first element contains the translation vector for moving the
        centroid into the origin.
        The second element contains the rotation matrix.
        The third element contains the translation vector for moving the
        structure onto the reference.
        The three transformations are performed sequentially.
    
    See Also
    --------
    apply_superimposition
    
    Notes
    -----
    The `transformation` tuple can be used in `apply_superimposition()` in
    order to transform another `AtomArray` in the same way. This can come in
    handy, in case you want to superimpose two structures with different
    amount of atoms. Often the two structures can be sliced in order to
    obtain the same size and annotation arrays. After superimposition the
    transformation can be applied on the original structure using
    `apply_superimposition()`.
    
    .. [1] W Kabsch, "A solution for the best rotation to relate two sets of
       vectors." Acta Cryst, 32, 922-923 (1976).
       
    .. [2] W Kabsch, "A discussion of the solution for the best rotation to
       relate two sets of vectors." Acta Cryst, 34, 827-828 (1978).
    """
    if type(reference) != AtomArray:
        raise ValueError("Reference must be AtomArray")
    if type(subject) == AtomArray:
        # Simply superimpose without loop
        return _superimpose(reference, subject, ca_only)
    elif type(subject) == AtomArrayStack:
        # Superimpose for every array in AtomArrayStack
        fitted_subjects = []
        transformations = []
        for array in subject:
            fitted_subject, transformation = _superimpose(reference,
                                                          array, ca_only)
            fitted_subjects.append(fitted_subject)
            transformations.append(transformation)
        # Convert AtomArray list back to AtomArrayStack
        fitted_subjects = stack(fitted_subjects)
        return (fitted_subjects, transformations)
    else:
        raise ValueError("Reference must be AtomArray")


def _superimpose(reference, subject, ca_only):
    """
    Performs the actual superimposition.
    
    See Also
    --------
    superimpose
    """
    if type(subject) == AtomArray:
        sub_centroid = centroid(subject)
        ref_centroid = centroid(reference)
        if ca_only:
            # For performance reasons the Kabsch algorithm
            # is only performed with "CA" per default
            # Implicitly this creates array copies
            sub_centered = subject[(subject.atom_name == " CA ")]
            ref_centered = reference[(reference.atom_name == " CA ")]
        else:
            sub_centered = subject.copy();
            ref_centered = reference.copy();
            
        if len(sub_centered) != len(ref_centered):
            raise BadStructureException("The subject and reference array have different amount of atoms")
        
        sub_centered.coord -= sub_centroid
        ref_centered.coord -= ref_centroid
        
        # Calculating rotation matrix using Kabsch algorithm
        y = sub_centered.coord
        x = ref_centered.coord
        # Calculate covariance matrix
        cov = np.dot(y.T, x)
        v, s, w = np.linalg.svd(cov)
        rotation = np.dot(w, v.T)
        # Remove pssibility of reflected atom coordinates
        if np.linalg.det(v) * np.linalg.det(w) < 0:
            s[-1,:] *= -1
            v[:,-1] *= -1
        rotation = np.dot(v,w)
        
        if ca_only:
            # Superimposed AtomArray should not be "CA" sequence
            # therefore original subject is copied 
            fitted_subject = subject.copy()
            fitted_subject.coord -= sub_centroid
        else:
            fitted_subject = sub_centered
        fitted_subject.coord = np.dot(fitted_subject.coord, rotation)
        fitted_subject.coord += ref_centroid
        return fitted_subject, (-sub_centroid,rotation,ref_centroid)


def apply_superimposition(atoms, transformation):
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