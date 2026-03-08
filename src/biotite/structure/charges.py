# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides one function for the computation of the partial
charges of the individual atoms of a given AtomArray according to the
PEOE algorithm of Gasteiger-Marsili.
"""

__name__ = "biotite.structure"
__author__ = "Jacob Marcel Anter, Patrick Kunzmann"
__all__ = ["partial_charges"]

import warnings
import numpy as np
from biotite.rust.structure import partial_charges as rust_partial_charges


def partial_charges(atom_array, iteration_step_num=6, charges=None):
    """
    partial_charges(atom_array, iteration_step_num=6, charges=None)

    Compute the partial charge of the individual atoms comprised in a
    given :class:`AtomArray` depending on their electronegativity.

    This function implements the
    *partial equalization of orbital electronegativity* (PEOE)
    algorithm :footcite:`Gasteiger1980`.

    Parameters
    ----------
    atom_array: AtomArray, shape=(n,)
        The :class:`AtomArray` to get the partial charge values for.
        Must have an associated `BondList`.
    iteration_step_num: int, optional
        The number of iteration steps is an optional argument and can be
        chosen by the user depending on the desired precision of the
        result. If no value is entered by the user, the default value
        ``6`` will be used.
        Gasteiger and Marsili described this number as sufficient.
    charges: ndarray, dtype=int, optional
        The array comprising the formal charges of the atoms in the
        input `atom_array`.
        If none is given, the ``charge`` annotation category of the
        input `atom_array` is used.
        If neither of them is given, the formal charges of all atoms
        will be arbitrarily set to zero.

    Returns
    -------
    charges: ndarray, dtype=float32
        The partial charge values of the individual atoms in the input
        `atom_array`.

    Notes
    -----
    A :class:`BondList` must be associated to the input
    :class:`AtomArray`.
    Otherwise, an error will be raised.
    Example:

    .. code-block:: python

        atom_array.bonds = struc.connect_via_residue_names(atom_array)

    |

    For the electronegativity of positively charged hydrogen, the
    value of 20.02 eV is used.

    Also note that the algorithm used in this function does not deliver
    proper results for expanded pi-electron systems like aromatic rings.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> fluoromethane = residue("CF0")
    >>> print(fluoromethane.atom_name)
    ['C1' 'F1' 'H1' 'H2' 'H3']
    >>> print(partial_charges(fluoromethane, iteration_step_num=1))
    [ 0.115 -0.175  0.020  0.020  0.020]
    >>> print(partial_charges(fluoromethane, iteration_step_num=6))
    [ 0.079 -0.253  0.058  0.058  0.058]
    """
    if atom_array.bonds is None:
        raise AttributeError(
            "The input AtomArray doesn't possess an associated BondList."
        )

    if charges is None:
        try:
            charges = atom_array.charge.astype(np.float32)
        except AttributeError:
            charges = np.zeros(atom_array.shape[0], dtype=np.float32)
            warnings.warn(
                "A charge array was neither given as optional "
                "argument, nor does a charge annotation of the "
                "inserted AtomArray exist. Therefore, all atoms' "
                "formal charge is assumed to be zero.",
                UserWarning,
            )
    else:
        charges = np.asarray(charges, dtype=np.float32).copy()

    if atom_array.bonds.get_bond_count() == 0:
        # No bonds along which charge transfer can occur
        # -> Partial charges equal formal charges (e.g. ions)
        return charges

    # Call Rust for hybridization determination, parameter lookup
    # and iterative charge transfer
    return np.asarray(
        rust_partial_charges(
            atom_array.element.tolist(),
            charges,
            atom_array.bonds,
            iteration_step_num,
        )
    )
