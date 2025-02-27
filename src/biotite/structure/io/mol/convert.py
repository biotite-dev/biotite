# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mol"
__author__ = "Patrick Kunzmann"
__all__ = ["get_structure", "set_structure"]

from biotite.structure.bonds import BondType
from biotite.structure.io.mol.mol import MOLFile
from biotite.structure.io.mol.sdf import SDFile, SDRecord


def get_structure(mol_file, record_name=None):
    """
    Get an :class:`AtomArray` from the MOL file.

    Ths function is a thin wrapper around
    :meth:`MOLFile.get_structure()`.

    Parameters
    ----------
    mol_file : MOLFile or SDFile or SDRecord
        The file.
    record_name : str, optional
        Has only an effect when `mol_file` is a :class:`SDFile`.
        The name of the record in the SD file.
        By default, the first record is used.

    Returns
    -------
    array : AtomArray
        This :class:`AtomArray` contains the optional ``charge``
        annotation and has an associated :class:`BondList`.
        All other annotation categories, except ``element`` are
        empty.
    """
    record = _get_record(mol_file, record_name)
    return record.get_structure()


def set_structure(
    mol_file, atoms, default_bond_type=BondType.ANY, version=None, record_name=None
):
    """
    Set the :class:`AtomArray` for the MOL file.

    Ths function is a thin wrapper around
    :meth:`MOLFile.set_structure()`.

    Parameters
    ----------
    mol_file : MOLFile
        The MOL file.
    atoms : AtomArray
        The array to be saved into this file.
        Must have an associated :class:`BondList`.
        Bond type fallback for the *Bond block*, if a
        :class:`BondType` has no CTAB counterpart.
        By default, each such bond is treated as
        :attr:`BondType.ANY`.
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
    record_name : str, optional
        Has only an effect when `mol_file` is a :class:`SDFile`.
        The name of the record.
        Default is the first record of the file.
        If the file is empty, a new record will be created.
    """
    record = _get_or_create_record(mol_file, record_name)
    record.set_structure(atoms, default_bond_type, version)


def _get_record(file, record_name):
    if isinstance(file, (MOLFile, SDRecord)):
        return file
    elif isinstance(file, SDFile):
        # Determine record
        if record_name is None:
            return file.record
        else:
            return file[record_name]
    else:
        raise TypeError(f"Unsupported file type '{type(file).__name__}'")


def _get_or_create_record(file, record_name):
    if isinstance(file, (MOLFile, SDRecord)):
        return file
    elif isinstance(file, SDFile):
        if record_name is None:
            if len(file) > 0:
                # Choose first record by default
                record_name = next(iter(file.keys()))
            else:
                # File is empty -> invent a new record name
                record_name = "Molecule"

        if record_name not in file:
            record = SDRecord()
            file[record_name] = record
        return file[record_name]
    else:
        raise TypeError(f"Unsupported file type '{type(file).__name__}'")
