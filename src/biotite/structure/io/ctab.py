# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io"
__author__ = "Patrick Kunzmann"
__all__ = ["read_structure_from_ctab", "write_structure_to_ctab"]

import warnings
from ..bonds import BondType


def read_structure_from_ctab(ctab_lines):
    """
    Parse a *MDL* connection table (Ctab) to obtain an
    :class:`AtomArray`. :footcite:`Dalby1992`.

    DEPRECATED: Moved to :mod:`biotite.structure.io.mol.ctab`.

    Parameters
    ----------
    ctab_lines : lines of str
        The lines containing the *ctab*.
        Must begin with the *counts* line and end with the `M END` line

    Returns
    -------
    atoms : AtomArray
        This :class:`AtomArray` contains the optional ``charge``
        annotation and has an associated :class:`BondList`.

    References
    ----------

    .. footbibliography::
    """
    warnings.warn("Moved to biotite.structure.io.mol.ctab", DeprecationWarning)
    from biotite.structure.io.mol.ctab import read_structure_from_ctab
    return read_structure_from_ctab(ctab_lines)


def write_structure_to_ctab(atoms, default_bond_type=BondType.ANY):
    """
    Convert an :class:`AtomArray` into a
    *MDL* connection table (Ctab). :footcite:`Dalby1992`

    DEPRECATED: Moved to :mod:`biotite.structure.io.mol.ctab`.

    Parameters
    ----------
    atoms : AtomArray
        The array must have an associated :class:`BondList`.

    Returns
    -------
    ctab_lines : lines of str
        The lines containing the *ctab*.
        The lines begin with the *counts* line and end with the `M END`
        .line
    default_bond_type : BondType
        Bond type fallback in the *Bond block* if a bond has no bond_type
        defined in *atoms* array. By default, each bond is treated as
        :attr:`BondType.ANY`.

    References
    ----------

    .. footbibliography::
    """
    warnings.warn("Moved to biotite.structure.io.mol.ctab", DeprecationWarning)
    from biotite.structure.io.mol.ctab import write_structure_to_ctab
    return write_structure_to_ctab(atoms, default_bond_type)
