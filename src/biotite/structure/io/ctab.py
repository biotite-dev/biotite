# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Functions for parsing and writing an :class:`AtomArray` from/to
*MDL* connection tables (Ctab).
"""

__name__ = "biotite.structure.io"
__author__ = "Patrick Kunzmann"
__all__ = ["read_structure_from_ctab", "write_structure_to_ctab"]

import warnings
import numpy as np
from biotite.structure.error import BadStructureError
from ..atoms import AtomArray, AtomArrayStack
from ..bonds import BondList, BondType

BOND_TYPE_MAPPING = {
    1: BondType.SINGLE,
    2: BondType.DOUBLE,
    3: BondType.TRIPLE,
    6: BondType.SINGLE,
    7: BondType.DOUBLE,
    8: BondType.ANY,
}
BOND_TYPE_MAPPING_REV = {
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC_SINGLE: 1,
    BondType.AROMATIC_DOUBLE: 2,
    BondType.ANY: 8,
}

CHARGE_MAPPING = {0: 0, 1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}
CHARGE_MAPPING_REV = {val: key for key, val in CHARGE_MAPPING.items()}


def read_structure_from_ctab(ctab_lines):
    """
    Parse a *MDL* connection table (Ctab) to obtain an
    :class:`AtomArray`. :footcite:`Dalby1992`

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
    n_atoms, n_bonds = _get_counts(ctab_lines[0])
    atom_lines = ctab_lines[1 : 1 + n_atoms]
    bond_lines = ctab_lines[1 + n_atoms : 1 + n_atoms + n_bonds]

    atoms = AtomArray(n_atoms)
    atoms.add_annotation("charge", int)
    for i, line in enumerate(atom_lines):
        atoms.coord[i, 0] = float(line[0:10])
        atoms.coord[i, 1] = float(line[10:20])
        atoms.coord[i, 2] = float(line[20:30])
        atoms.element[i] = line[31:34].strip().upper()
        charge = CHARGE_MAPPING.get(int(line[36:39]))
        if charge is None:
            warnings.warn(
                f"Cannot handle MDL charge type {int(line[36 : 39])}, "
                f"0 is used instead"
            )
            charge = 0
        atoms.charge[i] = charge

    bond_array = np.zeros((n_bonds, 3), dtype=np.uint32)
    for i, line in enumerate(bond_lines):
        bond_type = BOND_TYPE_MAPPING.get(int(line[6:9]))
        if bond_type is None:
            warnings.warn(
                f"Cannot handle MDL bond type {int(line[6 : 9])}, "
                f"BondType.ANY is used instead"
            )
            bond_type = BondType.ANY
        bond_array[i, 0] = int(line[0:3]) - 1
        bond_array[i, 1] = int(line[3:6]) - 1
        bond_array[i, 2] = bond_type
    atoms.bonds = BondList(n_atoms, bond_array)

    return atoms


def write_structure_to_ctab(atoms, default_bond_type=BondType.ANY):
    """
    Convert an :class:`AtomArray` into a
    *MDL* connection table (Ctab). :footcite:`Dalby1992`

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
    if isinstance(atoms, AtomArrayStack):
        raise TypeError(
            "An 'AtomArrayStack' was given, "
            "but only a single model can be written"
        )
    if atoms.bonds is None:
        raise BadStructureError("Input AtomArray has no associated BondList")

    try:
        charge = atoms.charge
    except AttributeError:
        charge = np.zeros(atoms.array_length(), dtype=int)

    atom_lines = [
        f"{atoms.coord[i,0]:>10.5f}"
        f"{atoms.coord[i,1]:>10.5f}"
        f"{atoms.coord[i,2]:>10.5f}"
        f" {atoms.element[i]:>3}"
        f"  {CHARGE_MAPPING_REV.get(charge[i], 0):>3d}" + f"{0:>3d}" * 10
        for i in range(atoms.array_length())
    ]

    default_bond_value = BOND_TYPE_MAPPING_REV[default_bond_type]

    bond_lines = [
        f"{i+1:>3d}{j+1:>3d}"
        f"{BOND_TYPE_MAPPING_REV.get(bond_type, default_bond_value):>3d}"
        + f"{0:>3d}" * 4
        for i, j, bond_type in atoms.bonds.as_array()
    ]

    counts_line = (
        f"{len(atom_lines):>3d}{len(bond_lines):>3d}     0  0  0  0  0  1 "
        "V2000"
    )

    return [counts_line] + atom_lines + bond_lines + ["M  END"]


def _get_counts(counts_line):
    elements = counts_line.split()
    return int(elements[0]), int(elements[1])
