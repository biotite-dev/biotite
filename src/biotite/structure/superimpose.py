# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for structure superimposition.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann, Claude J. Rogers"
__all__ = ["superimpose", "superimpose_apply"]

import numpy as np
from .atoms import coord
from .geometry import centroid


def superimpose(fixed, mobile, atom_mask=None):
    """
    Superimpose structures onto a fixed structure.
    
    The superimposition is performed using the Kabsch algorithm
    :footcite:`Kabsch1976, Kabsch1978`, so that the RMSD between the
    superimposed and the fixed structure is minimized.
    
    Parameters
    ----------
    fixed : AtomArray, shape(n,) or ndarray, shape(n,), dtype=float
        The fixed structure.
        Alternatively coordinates can be given.
    mobile: AtomArray, shape(n,) or AtomArrayStack, shape(m,n) or ndarray, shape(n,), dtype=float or ndarray, shape(m,n), dtype=float
        The structure(s) which is/are superimposed on the `fixed`
        structure.
        Each atom at index *i* in `mobile` must correspond the
        atom at index *i* in `fixed` to obtain correct results.
        Alternatively coordinates can be given.
    atom_mask: ndarray, dtype=bool, optional
        If given, only the atoms covered by this boolean mask will be
        considered for superimposition.
        This means that the algorithm will minimize the RMSD based
        on the covered atoms instead of all atoms.
        The returned superimposed structure will contain all atoms
        of the input structure, regardless of this parameter.
    
    Returns
    -------
    fitted : AtomArray or AtomArrayStack or ndarray, shape(n,), dtype=float or ndarray, shape(m,n), dtype=float
        A copy of the `mobile` structure(s),
        superimposed on the fixed structure.
        Only coordinates are returned, if coordinates were given in
        `mobile`.
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
    The `transformation` tuple can be used in
    :func:`superimpose_apply()` in order to transform another
    :class:`AtomArray` in the same way.
    This can come in handy, in case you want to superimpose two
    structures with different amount of atoms.
    Often the two structures need to be filtered in order to obtain the
    same size and annotation arrays.
    After superimposition the transformation can be applied on the
    original structure using :func:`superimpose_apply()`.
    
    References
    ----------
    
    .. footbibliography::
    
    Examples
    --------
    
    At first two models of a structure are taken and one of them is
    randomly rotated/translated.
    Consequently the RMSD is quite large:
    
    >>> array1 = atom_array_stack[0]
    >>> array2 = atom_array_stack[1]
    >>> array2 = translate(array2, [1,2,3])
    >>> array2 = rotate(array2, [1,2,3])
    >>> print("{:.3f}".format(rmsd(array1, array2)))
    11.260
    
    RMSD decreases after superimposition of only CA atoms:
    
    >>> array2_fit, transformation = superimpose(
    ...     array1, array2, atom_mask=(array2.atom_name == "CA")
    ... )
    >>> print("{:.3f}".format(rmsd(array1, array2_fit)))
    1.961

    RMSD is even lower when all atoms are considered in the
    superimposition:

    >>> array2_fit, transformation = superimpose(array1, array2)
    >>> print("{:.3f}".format(rmsd(array1, array2_fit)))
    1.928
    """

    m_coord = coord(mobile)
    f_coord = coord(fixed)
    mshape = m_coord.shape
    mdim = m_coord.ndim
    if f_coord.ndim != 2:
        raise ValueError("Expected fixed array to be an AtomArray")
    if mdim < 2:
        raise ValueError(
            "Expected mobile array to be an AtomArray or AtomArrayStack"
        )
    if mdim == 2:
        # normalize inputs. Fixed coords has shape (n, 3)
        # and mobile has shape (m, n, 3)
        m_coord = m_coord[np.newaxis, ...]

    nmodels = m_coord.shape[0]
    if f_coord.shape[0] != m_coord.shape[1]:
        raise ValueError(
            f"Expected fixed array and mobile array to have the same number "
            f"of atoms, but {f_coord.shape[0]} != {m_coord.shape[1]}"
        )

    if atom_mask is not None:
        # Implicitly this creates array copies
        mob_filtered = m_coord[..., atom_mask, :]
        fix_filtered = f_coord[atom_mask, :]
    else:
        mob_filtered = np.copy(m_coord)
        fix_filtered = np.copy(f_coord)
    
    # Center coordinates at (0,0,0)
    mob_centroid = centroid(mob_filtered)
    fix_centroid = centroid(fix_filtered)
    mob_filtered -= mob_centroid[..., np.newaxis, :]
    fix_filtered -= fix_centroid
    
    s_coord = m_coord.copy() - mob_centroid[..., np.newaxis, :]
    # Perform Kabsch algorithm for every model
    transformations = [None] * nmodels
    for i in range(nmodels):
        rotation = _superimpose(fix_filtered, mob_filtered[i])
        s_coord[i] = np.dot(rotation, s_coord[i].T).T
        transformations[i] = (-mob_centroid[i], rotation, fix_centroid)
    s_coord += fix_centroid
    
    if isinstance(mobile, np.ndarray):
        superimposed = s_coord.reshape(mshape)
    else:
        superimposed = mobile.copy()
        superimposed.coord = s_coord.reshape(mshape)

    if mdim == 2:
        return superimposed, transformations[0]
    else:
        return superimposed, transformations


def _superimpose(fix_centered, mob_centered):
    """
    Perform the Kabsch algorithm using only the coordinates.
    """
    # Calculating rotation matrix
    y = mob_centered
    x = fix_centered
    # Calculate covariance matrix
    cov = np.dot(x.T, y)
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
    atoms : AtomArray or ndarray, shape(n,), dtype=float
        The structure to apply the transformation on.
        Alternatively coordinates can be given.
    transformation: tuple, size=3
        The transformation tuple, obtained by :func:`superimpose()`.
    
    Returns
    -------
    fitted : AtomArray or AtomArrayStack
        A copy of the `atoms` structure,
        with transformations applied.
        Only coordinates are returned, if coordinates were given in
        `atoms`.
    
    See Also
    --------
    superimpose
    """
    trans1, rot, trans2 = transformation
    s_coord = coord(atoms).copy()
    s_coord += trans1
    s_coord = np.dot(rot, s_coord.T).T
    s_coord += trans2

    if isinstance(atoms, np.ndarray):
        return s_coord
    else:
        transformed = atoms.copy()
        transformed.coord = s_coord
        return transformed