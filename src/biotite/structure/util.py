# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Utility functions for in internal use in `Bio.Structure` package
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["vector_dot", "norm_vector", "distance", "matrix_rotate"]

import numpy as np
from .atoms import Atom, array


def vector_dot(v1,v2):
    """
    Calculate vector dot product of two vectors.
    
    Parameters
    ----------
    v1,v2 : ndarray
        The arrays to calculate the product from.
        The vectors are represented by the last axis.
    
    Returns
    -------
    product : float or ndarray
        Scalar product over the last dimension of the arrays.
    """
    return (v1*v2).sum(axis=-1)


def norm_vector(v):
    """
    Normalise a vector.
    
    Parameters
    ----------
    v : ndarray
        The array containg the vector(s).
        The vectors are represented by the last axis.
    """
    factor = np.linalg.norm(v, axis=-1)
    if isinstance(factor, np.ndarray):
        v /= factor[..., np.newaxis]
    else:
        v /= factor
        

def distance(v1,v2):
    """
    Calculate the distance between two position vectors.
    
    Parameters
    ----------
    v1,v2 : ndarray
        The arrays to calculate the product from.
        The vectors are represented by the last axis.
    
    Returns
    -------
    product : float or ndarray
        Vector distance over the last dimension of the array.
    """
    dif = v1 - v2
    return np.sqrt((dif*dif).sum(axis=-1))


def matrix_rotate(v, matrix):
    """
    Perform a rotation using a rotation matrix.

    Parameters
    ----------
    v : ndarray
        The coordinates to rotate.
    matrix : ndarray
        The rotation matrix.
    
    Returns
    -------
    rotated : ndarray
        The rotated coordinates.
    """
    # For proper rotation reshape into a maximum of 2 dimensions
    orig_ndim = v.ndim
    if orig_ndim > 2:
        orig_shape = v.shape
        v = v.reshape(-1, 3)
    # Apply rotation
    v = np.dot(matrix, v.T).T
    # Reshape back into original shape
    if orig_ndim > 2:
        v = v.reshape(*orig_shape)
    return v

def get_std_adenine():
    atom1 = Atom([-2.479, 5.346, 0.000], atom_name="C1′", res_name="A")
    atom2 = Atom([-1.291, 4.498, 0.000], atom_name="N9", res_name="A")
    atom3 = Atom([0.024, 4.897, 0.000], atom_name="C8", res_name="A")
    atom4 = Atom([0.877, 3.902, 0.000], atom_name="N7", res_name="A")
    atom5 = Atom([0.071, 2.771, 0.000], atom_name="C5", res_name="A")
    atom6 = Atom([0.369, 1.398, 0.000], atom_name="C6", res_name="A")
    atom7 = Atom([1.611, 0.909, 0.000], atom_name="N6", res_name="A")
    atom8 = Atom([-0.668, 0.532, 0.000], atom_name="N1", res_name="A")
    atom9 = Atom([-1.912, 1.023, 0.000], atom_name="C2", res_name="A")
    atom10 = Atom([-2.320, 2.290, 0.000], atom_name="N3", res_name="A")
    atom11 = Atom([-1.267, 3.124, 0.000], atom_name="C4", res_name="A")

    pyrimidine_center = np.mean([atom5.coord, atom6.coord, atom8.coord,
                                    atom9.coord, atom10.coord, atom11.coord],
                                    axis=-2
                            )
    imidazole_center = np.mean([atom2.coord, atom3.coord, atom4.coord,
                                    atom5.coord, atom11.coord], 
                                    axis=-2
                            )

    adenine = array([atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, 
                        atom9, atom10, atom11]
                )

    return adenine, [pyrimidine_center, imidazole_center]

def get_std_cytosine():
    atom1 = Atom([-2.477, 5.402, 0.000], atom_name="C1′", res_name="C")
    atom2 = Atom([-1.285, 4.542, 0.000], atom_name="N1", res_name="C")
    atom3 = Atom([-1.472, 3.158, 0.000], atom_name="C2", res_name="C")
    atom4 = Atom([-2.628, 2.709, 0.000], atom_name="O2", res_name="C")
    atom5 = Atom([-0.391, 2.344, 0.000], atom_name="N3", res_name="C")
    atom6 = Atom([0.837, 2.868, 0.000], atom_name="C4", res_name="C")
    atom7 = Atom([1.875, 2.027, 0.000], atom_name="N4", res_name="C")
    atom8 = Atom([1.056, 4.275, 0.000], atom_name="C5", res_name="C")
    atom9 = Atom([-0.023, 5.068, 0.000], atom_name="C6", res_name="C")

    pyrimidine_center = np.mean([atom2.coord, atom3.coord, atom5.coord,
                                    atom6.coord, atom8.coord, atom9.coord],
                                    axis=-2
                            )

    cytosine = array([atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, 
                        atom9]
                    )

    return cytosine, [pyrimidine_center]

def get_std_guanine():
    atom1 = Atom([-2.477, 5.399, 0.000], atom_name="C1′", res_name="G")
    atom2 = Atom([-1.289, 4.551, 0.000], atom_name="N9", res_name="G")
    atom3 = Atom([0.023, 4.962, 0.000], atom_name="C8", res_name="G")
    atom4 = Atom([0.870, 3.969, 0.000], atom_name="N7", res_name="G")
    atom5 = Atom([0.071, 2.833, 0.000], atom_name="C5", res_name="G")
    atom6 = Atom([0.424, 1.460, 0.000], atom_name="C6", res_name="G")
    atom7 = Atom([1.554, 0.955, 0.000], atom_name="O6", res_name="G")
    atom8 = Atom([-0.700, 0.641, 0.000], atom_name="N1", res_name="G")
    atom9 = Atom([-1.999, 1.087, 0.000], atom_name="C2", res_name="G")
    atom10 = Atom([-2.949, 0.139, -0.001], atom_name="N2", res_name="G")
    atom11 = Atom([-2.342, 2.364, 0.001], atom_name="N3", res_name="G")
    atom12 = Atom([-1.265, 3.177, 0.000], atom_name="C4", res_name="G")

    pyrimidine_center = np.mean([atom5.coord, atom6.coord, atom8.coord,
                                    atom9.coord, atom11.coord, atom12.coord],
                                    axis=-2
                                )
    imidazole_center = np.mean([atom2.coord, atom3.coord, atom4.coord,
                                    atom5.coord, atom12.coord],
                                    axis=-2
                                )

    guanine = array([atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, 
                        atom9, atom10, atom11, atom12]
                )

    return guanine, [pyrimidine_center, imidazole_center]

def get_std_thymine():
    atom1 = Atom([-2.481, 5.354, 0.000], atom_name="C1′", res_name="T")
    atom2 = Atom([-1.284, 4.500, 0.000], atom_name="N1", res_name="T")
    atom3 = Atom([-1.462, 3.135, 0.000], atom_name="C2", res_name="T")
    atom4 = Atom([-2.562, 2.608, 0.000], atom_name="O2", res_name="T")
    atom5 = Atom([-0.298, 2.407, 0.000], atom_name="N3", res_name="T")
    atom6 = Atom([0.994, 2.897, 0.000], atom_name="C4", res_name="T")
    atom7 = Atom([1.944, 2.119, 0.000], atom_name="O4", res_name="T")
    atom8 = Atom([1.106, 4.338, 0.000], atom_name="C5", res_name="T")
    atom9 = Atom([2.466, 4.961, 0.001], atom_name="C5M", res_name="T")
    atom10 = Atom([-0.024, 5.057, 0.000], atom_name="C6", res_name="T")

    pyrimidine_center = np.mean([atom2.coord, atom3.coord, atom5.coord,
                                    atom6.coord, atom8.coord, atom10.coord],
                                    axis=-2
                            )

    thymine = array([atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, 
                        atom9, atom10]
                )

    return thymine, [pyrimidine_center]

def get_std_uracil():
    atom1 = Atom([-2.481, 5.354, 0.000], atom_name="C1′", res_name="U")
    atom2 = Atom([-1.284, 4.500, 0.000], atom_name="N1", res_name="U")
    atom3 = Atom([-1.462, 3.131, 0.000], atom_name="C2", res_name="U")
    atom4 = Atom([-2.563, 2.608, 0.000], atom_name="O2", res_name="U")
    atom5 = Atom([-0.302, 2.397, 0.000], atom_name="N3", res_name="U")
    atom6 = Atom([0.989, 2.884, 0.000], atom_name="C4", res_name="U")
    atom7 = Atom([1.935, 2.094, -0.001], atom_name="O4", res_name="U")
    atom8 = Atom([1.089, 4.311, 0.000], atom_name="C5", res_name="U")
    atom9 = Atom([-0.024, 5.053, 0.000], atom_name="C6", res_name="U")
    
    pyrimidine_center = np.mean([atom2.coord, atom3.coord, atom5.coord,
                                    atom6.coord, atom8.coord, atom9.coord],
                                    axis=-2
                            )
    
    uracil = array([atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, 
                        atom9]
                )
                 
    return uracil



