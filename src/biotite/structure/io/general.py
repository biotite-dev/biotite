# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module contains a convenience function for loading structures from
general structure files.
"""

__name__ = "biotite.structure.io"
__author__ = "Patrick Kunzmann"
__all__ = ["load_structure", "save_structure"]

import datetime
import io
import os.path
from biotite.structure.atoms import AtomArrayStack


def load_structure(file_path, template=None, **kwargs):
    """
    Load an :class:`AtomArray` or class`AtomArrayStack` from a structure
    file without the need to manually instantiate a :class:`File`
    object.

    Internally this function uses a :class:`File` object, based on the
    file extension.
    Trajectory files furthermore require specification of the `template`
    parameter.

    Parameters
    ----------
    file_path : str
        The path to structure file.
    template : AtomArray or AtomArrayStack or file-like object or str, optional
        Only required when reading a trajectory file.
    **kwargs
        Additional parameters will be passed to either the
        :func:`get_structure()` or :func:`read()` method of the file
        object.
        This does not affect files given via the `template` parameter.
        The only exception is the `atom_i`, which is applied to the template
        as well if number of atoms do not match.

    Returns
    -------
    array : AtomArray or AtomArrayStack
        If the file contains multiple models, an AtomArrayStack is
        returned, otherwise an AtomArray is returned.

    Raises
    ------
    ValueError
        If the file format (i.e. the file extension) is unknown.
    TypeError
        If a trajectory file is loaded without specifying the
        `template` parameter.
    """
    # Optionally load template from file
    if isinstance(template, (io.IOBase, str)):
        template = load_structure(template)

    # We only need the suffix here
    _, suffix = os.path.splitext(file_path)
    match suffix:
        case ".pdb":
            from biotite.structure.io.pdb import PDBFile

            file = PDBFile.read(file_path)
            array = file.get_structure(**kwargs)
            return _as_single_model_if_possible(array)
        case ".pdbqt":
            from biotite.structure.io.pdbqt import PDBQTFile

            file = PDBQTFile.read(file_path)
            array = file.get_structure(**kwargs)
            return _as_single_model_if_possible(array)
        case ".cif" | ".pdbx":
            from biotite.structure.io.pdbx import CIFFile, get_structure

            file = CIFFile.read(file_path)
            array = get_structure(file, **kwargs)
            return _as_single_model_if_possible(array)
        case ".bcif":
            from biotite.structure.io.pdbx import BinaryCIFFile, get_structure

            file = BinaryCIFFile.read(file_path)
            array = get_structure(file, **kwargs)
            return _as_single_model_if_possible(array)
        case ".gro":
            from biotite.structure.io.gro import GROFile

            file = GROFile.read(file_path)
            array = file.get_structure(**kwargs)
            return _as_single_model_if_possible(array)
        case ".mol":
            from biotite.structure.io.mol import MOLFile

            file = MOLFile.read(file_path)
            array = file.get_structure(**kwargs)
            # MOL and SDF files only contain a single model
            return array
        case ".sdf" | ".sd":
            from biotite.structure.io.mol import SDFile, get_structure

            file = SDFile.read(file_path)
            array = get_structure(file, **kwargs)
            return array
        case ".trr" | ".xtc" | ".dcd" | ".netcdf":
            if template is None:
                raise TypeError("Template must be specified for trajectory files")
            # Filter template for atom ids, if an unfiltered template
            if "atom_i" in kwargs and template.shape[-1] != len(kwargs["atom_i"]):
                template = template[..., kwargs["atom_i"]]
            from biotite.structure.io.dcd import DCDFile
            from biotite.structure.io.netcdf import NetCDFFile
            from biotite.structure.io.trr import TRRFile
            from biotite.structure.io.xtc import XTCFile

            if suffix == ".trr":
                traj_file_cls = TRRFile
            if suffix == ".xtc":
                traj_file_cls = XTCFile
            if suffix == ".dcd":
                traj_file_cls = DCDFile
            if suffix == ".netcdf":
                traj_file_cls = NetCDFFile
            file = traj_file_cls.read(file_path, **kwargs)
            return file.get_structure(template)
        case unknown_suffix:
            raise ValueError(f"Unknown file format '{unknown_suffix}'")


def save_structure(file_path, array, **kwargs):
    """
    Save an :class:`AtomArray` or class`AtomArrayStack` to a structure
    file without the need to manually instantiate a :class:`File`
    object.

    Internally this function uses a :class:`File` object, based on the
    file extension.

    Parameters
    ----------
    file_path : str
        The path to structure file.
    array : AtomArray or AtomArrayStack
        The structure to be saved.
    **kwargs
        Additional parameters will be passed to the respective `set_structure`
        method.

    Raises
    ------
    ValueError
        If the file format (i.e. the file extension) is unknown.
    """
    # We only need the suffix here
    _, suffix = os.path.splitext(file_path)
    match suffix:
        case ".pdb":
            from biotite.structure.io.pdb import PDBFile

            file = PDBFile()
            file.set_structure(array, **kwargs)
            file.write(file_path)
        case ".pdbqt":
            from biotite.structure.io.pdbqt import PDBQTFile

            file = PDBQTFile()
            file.set_structure(array, **kwargs)
            file.write(file_path)
        case ".cif" | ".pdbx":
            from biotite.structure.io.pdbx import CIFFile, set_structure

            file = CIFFile()
            set_structure(file, array, **kwargs)
            file.write(file_path)
        case ".bcif":
            from biotite.structure.io.pdbx import BinaryCIFFile, set_structure

            file = BinaryCIFFile()
            set_structure(file, array, **kwargs)
            file.write(file_path)
        case ".gro":
            from biotite.structure.io.gro import GROFile

            file = GROFile()
            file.set_structure(array, **kwargs)
            file.write(file_path)
        case ".mol":
            from biotite.structure.io.mol import MOLFile

            file = MOLFile()
            file.set_structure(array, **kwargs)
            file.header = _mol_header()
            file.write(file_path)
        case ".sdf" | ".sd":
            from biotite.structure.io.mol import SDFile, SDRecord, set_structure

            record = SDRecord()
            record.set_structure(array, **kwargs)
            record.header = _mol_header()
            file = SDFile({"Molecule": record})
            file.write(file_path)
        case ".trr" | ".xtc" | ".dcd" | ".netcdf":
            from biotite.structure.io.dcd import DCDFile
            from biotite.structure.io.netcdf import NetCDFFile
            from biotite.structure.io.trr import TRRFile
            from biotite.structure.io.xtc import XTCFile

            if suffix == ".trr":
                traj_file_cls = TRRFile
            if suffix == ".xtc":
                traj_file_cls = XTCFile
            if suffix == ".dcd":
                traj_file_cls = DCDFile
            if suffix == ".netcdf":
                traj_file_cls = NetCDFFile
            file = traj_file_cls()
            file.set_structure(array, **kwargs)
            file.write(file_path)
        case unknown_suffix:
            raise ValueError(f"Unknown file format '{unknown_suffix}'")


def _as_single_model_if_possible(atoms):
    if isinstance(atoms, AtomArrayStack) and atoms.stack_depth() == 1:
        # Stack containing only one model -> return as atom array
        return atoms[0]
    else:
        return atoms


def _mol_header():
    from biotite.structure.io.mol import Header

    return Header(
        mol_name="Molecule",
        program="Biotite",
        time=datetime.datetime.now(),
        dimensions="3D",
    )
