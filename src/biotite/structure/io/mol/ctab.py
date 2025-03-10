# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Functions for parsing and writing an :class:`AtomArray` from/to
*MDL* connection tables (Ctab).
"""

__name__ = "biotite.structure.io.mol"
__author__ = "Patrick Kunzmann"
__all__ = ["read_structure_from_ctab", "write_structure_to_ctab"]

import itertools
import shlex
import warnings
import numpy as np
from biotite.file import InvalidFileError
from biotite.structure.atoms import AtomArray, AtomArrayStack
from biotite.structure.bonds import BondList, BondType
from biotite.structure.error import BadStructureError
from biotite.structure.io.util import number_of_integer_digits

BOND_TYPE_MAPPING = {
    1: BondType.SINGLE,
    2: BondType.DOUBLE,
    3: BondType.TRIPLE,
    4: BondType.AROMATIC,
    5: BondType.ANY,
    6: BondType.AROMATIC_SINGLE,
    7: BondType.AROMATIC_DOUBLE,
    8: BondType.ANY,
}
BOND_TYPE_MAPPING_REV = {v: k for k, v in BOND_TYPE_MAPPING.items()}

CHARGE_MAPPING = {0: 0, 1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}
CHARGE_MAPPING_REV = {val: key for key, val in CHARGE_MAPPING.items()}

V2000_COMPATIBILITY_LINE = "  0  0  0  0  0  0  0  0  0  0999 V3000"
# The number of charges per `M  CHG` line
N_CHARGES_PER_LINE = 8


def read_structure_from_ctab(ctab_lines):
    """
    Parse a *MDL* connection table (Ctab) to obtain an
    :class:`AtomArray`.
    :footcite:`Dalby1992`

    Parameters
    ----------
    ctab_lines : lines of str
        The lines containing the *ctab*.
        Must begin with the *counts* line and end with the `M END` line.

    Returns
    -------
    atoms : AtomArray
        This :class:`AtomArray` contains the optional ``charge``
        annotation and has an associated :class:`BondList`.

    References
    ----------

    ``V3000`` specification was taken from
    `<https://discover.3ds.com/sites/default/files/2020-08/biovia_ctfileformats_2020.pdf>`_.

    .. footbibliography::
    """
    match _get_version(ctab_lines[0]):
        case "V2000":
            return _read_structure_from_ctab_v2000(ctab_lines)
        case "V3000":
            return _read_structure_from_ctab_v3000(ctab_lines)
        case "":
            raise InvalidFileError("CTAB counts line misses version")
        case unkown_version:
            raise InvalidFileError(f"Unknown CTAB version '{unkown_version}'")


def write_structure_to_ctab(atoms, default_bond_type=BondType.ANY, version=None):
    """
    Convert an :class:`AtomArray` into a
    *MDL* connection table (Ctab).
    :footcite:`Dalby1992`

    Parameters
    ----------
    atoms : AtomArray
        The array must have an associated :class:`BondList`.
    default_bond_type : BondType, optional
        Bond type fallback for the *Bond block*, if a :class:`BondType`
        has no CTAB counterpart.
        By default, each such bond is treated as :attr:`BondType.ANY`.
    version : {"V2000", "V3000"}, optional
        The version of the CTAB format.
        ``"V2000"`` uses the *Atom* and *Bond* block, while ``"V3000"``
        uses the *Properties* block.
        By default, ``"V2000"`` is used, unless the number of atoms or
        bonds exceeds 999, in which case ``"V3000"`` is used.

    Returns
    -------
    ctab_lines : lines of str
        The lines containing the *ctab*.
        The lines begin with the *counts* line and end with the `M END`
        line.

    References
    ----------

    ``V3000`` specification was taken from
    `<https://discover.3ds.com/sites/default/files/2020-08/biovia_ctfileformats_2020.pdf>`_.

    .. footbibliography::
    """
    if isinstance(atoms, AtomArrayStack):
        raise TypeError(
            "An 'AtomArrayStack' was given, but only a single model can be written"
        )
    if atoms.bonds is None:
        raise BadStructureError("Input AtomArray has no associated BondList")
    if np.isnan(atoms.coord).any():
        raise BadStructureError("Input AtomArray has NaN coordinates")

    match version:
        case None:
            if _is_v2000_compatible(atoms.array_length(), atoms.bonds.get_bond_count()):
                return _write_structure_to_ctab_v2000(atoms, default_bond_type)
            else:
                return _write_structure_to_ctab_v3000(atoms, default_bond_type)
        case "V2000":
            if not _is_v2000_compatible(
                atoms.array_length(), atoms.bonds.get_bond_count()
            ):
                raise ValueError(
                    "The given number of atoms or bonds is too large for V2000 format"
                )
            return _write_structure_to_ctab_v2000(atoms, default_bond_type)
        case "V3000":
            return _write_structure_to_ctab_v3000(atoms, default_bond_type)
        case unkown_version:
            raise ValueError(f"Unknown CTAB version '{unkown_version}'")


def _read_structure_from_ctab_v2000(ctab_lines):
    n_atoms, n_bonds = _get_counts_v2000(ctab_lines[0])
    atom_lines = ctab_lines[1 : 1 + n_atoms]
    bond_lines = ctab_lines[1 + n_atoms : 1 + n_atoms + n_bonds]
    charge_lines = [
        line
        for line in ctab_lines[1 + n_atoms + n_bonds :]
        if line.startswith("M  CHG")
    ]

    atoms = AtomArray(n_atoms)
    atoms.add_annotation("charge", int)
    for i, line in enumerate(atom_lines):
        atoms.coord[i, 0] = float(line[0:10])
        atoms.coord[i, 1] = float(line[10:20])
        atoms.coord[i, 2] = float(line[20:30])
        atoms.element[i] = line[31:34].strip().upper()
        # If one 'M CHG' entry is present,
        # it supersedes all atom charges in the atom block
        if not charge_lines:
            charge = CHARGE_MAPPING.get(int(line[36:39]))
            if charge is None:
                warnings.warn(
                    f"Cannot handle MDL charge type {int(line[36:39])}, "
                    f"0 is used instead"
                )
                charge = 0
            atoms.charge[i] = charge

    for line in charge_lines:
        # Remove 'M  CHGnn8' prefix
        line = line[9:]
        # The lines contains atom index and charge alternatingly
        for atom_i_str, charge_str in _batched(line.split(), 2):
            atom_index = int(atom_i_str) - 1
            charge = int(charge_str)
            atoms.charge[atom_index] = charge

    bond_array = np.zeros((n_bonds, 3), dtype=np.uint32)
    for i, line in enumerate(bond_lines):
        bond_type = BOND_TYPE_MAPPING.get(int(line[6:9]))
        if bond_type is None:
            warnings.warn(
                f"Cannot handle MDL bond type {int(line[6:9])}, "
                f"BondType.ANY is used instead"
            )
            bond_type = BondType.ANY
        bond_array[i, 0] = int(line[0:3]) - 1
        bond_array[i, 1] = int(line[3:6]) - 1
        bond_array[i, 2] = bond_type
    atoms.bonds = BondList(n_atoms, bond_array)

    return atoms


def _read_structure_from_ctab_v3000(ctab_lines):
    v30_lines = [line[6:].strip() for line in ctab_lines if line.startswith("M  V30")]

    atom_lines = _get_block_v3000(v30_lines, "ATOM")
    if len(atom_lines) == 0:
        raise InvalidFileError("ATOM block is empty")
    atoms = AtomArray(len(atom_lines))
    atoms.add_annotation("charge", int)
    # The V3000 atom index does not necessarily count from 1 to n,
    # but allows arbitrary positive integers
    # Hence, a mapping from V3000 atom index to AtomArray index is
    # needed to get the correct index for a bond
    v30_atom_indices = {}
    for i, line in enumerate(atom_lines):
        if "'" in line or '"' in line:
            columns = shlex.split(line)
        else:
            columns = line.split()
        v30_index = int(columns[0])
        v30_type = columns[1]
        if v30_type == "R#":
            raise NotImplementedError("Rgroup atoms are not supported")
        v30_coord = np.array(columns[2:5], dtype=float)
        v30_properties = create_property_dict_v3000(columns[6:])

        v30_atom_indices[v30_index] = i
        atoms.coord[i] = v30_coord
        atoms.element[i] = v30_type.upper()
        atoms.charge[i] = int(v30_properties.get("CHG", 0))

    bond_lines = _get_block_v3000(v30_lines, "BOND")
    bond_array = np.zeros((len(bond_lines), 3), dtype=np.uint32)
    for i, line in enumerate(bond_lines):
        columns = line.split()
        v30_type = int(columns[1])
        v30_atom_index_1 = int(columns[2])
        v30_atom_index_2 = int(columns[3])

        bond_type = BOND_TYPE_MAPPING.get(v30_type)
        if bond_type is None:
            warnings.warn(
                f"Cannot handle MDL bond type {v30_type}, BondType.ANY is used instead"
            )
            bond_type = BondType.ANY
        bond_array[i, 0] = v30_atom_indices[v30_atom_index_1]
        bond_array[i, 1] = v30_atom_indices[v30_atom_index_2]
        bond_array[i, 2] = bond_type
    atoms.bonds = BondList(atoms.array_length(), bond_array)

    return atoms


def _get_version(counts_line):
    return counts_line[33:39].strip()


def _is_v2000_compatible(n_atoms, n_bonds):
    # The format uses a maximum of 3 digits for the atom and bond count
    return n_atoms < 1000 and n_bonds < 1000


def _get_counts_v2000(counts_line):
    return int(counts_line[0:3]), int(counts_line[3:6])


def _get_block_v3000(v30_lines, block_name):
    block_lines = []
    in_block = False
    for line in v30_lines:
        if line.startswith(f"BEGIN {block_name}"):
            in_block = True
        elif line.startswith(f"END {block_name}"):
            if in_block:
                return block_lines
            else:
                raise InvalidFileError(f"Block '{block_name}' ended before it began")
        elif in_block:
            block_lines.append(line)
    return block_lines


def create_property_dict_v3000(property_strings):
    properties = {}
    for prop in property_strings:
        key, value = prop.split("=")
        properties[key] = value
    return properties


def _write_structure_to_ctab_v2000(atoms, default_bond_type):
    try:
        charge = atoms.charge
    except AttributeError:
        charge = np.zeros(atoms.array_length(), dtype=int)

    counts_line = (
        f"{atoms.array_length():>3d}{atoms.bonds.get_bond_count():>3d}"
        "  0     0  0  0  0  0  0  1 V2000"
    )

    for i, coord_name in enumerate(["x", "y", "z"]):
        n_coord_digits = number_of_integer_digits(atoms.coord[:, i])
        if n_coord_digits > 5:
            raise BadStructureError(
                f"5 pre-decimal columns for {coord_name}-coordinates are "
                f"available, but array would require {n_coord_digits}"
            )
    atom_lines = [
        f"{atoms.coord[i, 0]:>10.4f}"
        f"{atoms.coord[i, 1]:>10.4f}"
        f"{atoms.coord[i, 2]:>10.4f}"
        f" {atoms.element[i].capitalize():3}"
        f"{0:>2}"  # Mass difference -> unused
        f"{CHARGE_MAPPING_REV.get(charge[i], 0):>3d}"
        + f"{0:>3d}"
        * 10  # More unused fields
        for i in range(atoms.array_length())
    ]

    default_bond_value = BOND_TYPE_MAPPING_REV[default_bond_type]
    bond_lines = [
        f"{i + 1:>3d}{j + 1:>3d}"
        f"{BOND_TYPE_MAPPING_REV.get(bond_type, default_bond_value):>3d}"
        + f"{0:>3d}"
        * 4
        for i, j, bond_type in atoms.bonds.as_array()
    ]

    # V2000 files introduce charge annotations in the property block
    # They define the charge literally (without mapping)
    charge_lines = []
    # Each `M  CHG` line can contain up to 8 charges
    for batch in _batched(
        [(atom_i, c) for atom_i, c in enumerate(charge) if c != 0], N_CHARGES_PER_LINE
    ):
        charge_lines.append(
            f"M  CHG{len(batch):>3d}"
            + "".join(f" {atom_i + 1:>3d} {c:>3d}" for atom_i, c in batch)
        )

    return [counts_line] + atom_lines + bond_lines + charge_lines + ["M  END"]


def _write_structure_to_ctab_v3000(atoms, default_bond_type):
    try:
        charges = atoms.charge
    except AttributeError:
        charges = np.zeros(atoms.array_length(), dtype=int)

    counts_line = f"COUNTS {atoms.array_length()} {atoms.bonds.get_bond_count()} 0 0 0"

    for i, coord_name in enumerate(["x", "y", "z"]):
        n_coord_digits = number_of_integer_digits(atoms.coord[:, i])
        if n_coord_digits > 5:
            raise BadStructureError(
                f"5 pre-decimal columns for {coord_name}-coordinates are "
                f"available, but array would require {n_coord_digits}"
            )
    atom_lines = [
        f"{i + 1}"
        f" {_quote(atoms.element[i].capitalize())}"
        f" {atoms.coord[i, 0]:.4f}"
        f" {atoms.coord[i, 1]:.4f}"
        f" {atoms.coord[i, 2]:.4f}"
        # 'aamap' is unused
        f" 0"
        f" {_to_property(charges[i])}"
        for i in range(atoms.array_length())
    ]

    default_bond_value = BOND_TYPE_MAPPING_REV[default_bond_type]
    bond_lines = [
        f"{k + 1}"
        f" {BOND_TYPE_MAPPING_REV.get(bond_type, default_bond_value)}"
        f" {i + 1}"
        f" {j + 1}"
        for k, (i, j, bond_type) in enumerate(atoms.bonds.as_array())
    ]

    lines = (
        ["BEGIN CTAB"]
        + [counts_line]
        + ["BEGIN ATOM"]
        + atom_lines
        + ["END ATOM"]
        + ["BEGIN BOND"]
        + bond_lines
        + ["END BOND"]
        + ["END CTAB"]
    )
    # Mark lines as V3000 CTAB
    lines = ["M  V30 " + line for line in lines]
    return [V2000_COMPATIBILITY_LINE] + lines + ["M  END"]


def _to_property(charge):
    if charge == 0:
        return ""
    else:
        return f"CHG={charge}"


def _quote(string):
    if " " in string or len(string) == 0:
        return f'"{string}"'
    else:
        return string


def _batched(iterable, n):
    """
    Equivalent to :func:`itertools.batched()`.

    However, :func:`itertools.batched()` is available since Python 3.12.
    This function can be removed when the minimum supported Python
    version is 3.12.
    """
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch
