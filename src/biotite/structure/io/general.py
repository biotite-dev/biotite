# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

"""
This module contains a convenience function for loading structures from
general structure files.
"""

import os.path
from ..atoms import AtomArray, AtomArrayStack

__all__ = ["get_structure_from"]


def get_structure_from(file_path, template=None):
    """
    Load an atom array or stack from a structure file without the need
    to manually instantiate a `File` object.
    
    Internally this function uses a `File` object, based on the file
    extension. Trajectory files furthermore require specification of
    the `template` parameter
    
    Parameters
    ----------
    file_path : str
        The path to structure file.
    template : AtomArray or AtomArrayStack, optional
        Only required when reading a trajectory file.
    
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
    # We only need the suffix here
    filename, suffix = os.path.splitext(file_path)
    if suffix == ".pdb":
        from .pdb import PDBFile
        file = PDBFile()
        file.read(file_path)
        return file.get_structure()
    elif suffix == ".cif" or suffix == ".pdbx":
        from .pdbx import PDBxFile, get_structure
        file = PDBxFile()
        file.read(file_path)
        array = get_structure(file)
        if isinstance(array, AtomArrayStack) and array.stack_depth() == 1:
            # Stack containing only one model -> return as atom array
            return array[0]
        else:
            return array
    elif suffix == ".mmtf":
        raise NotImplementedError()
    elif suffix == ".npz":
        from .npz import NpzFile
        file = NpzFile()
        file.read(file_path)
        array = file.get_structure()
        if isinstance(array, AtomArrayStack) and array.stack_depth() == 1:
            # Stack containing only one model -> return as atom array
            return array[0]
        else:
            return array
    elif suffix == "trr":
        if template is None:
            raise TypeError("Template must be specified for trajectory files")
        from .trr import TRRFile
        file = TRRFile()
        file.read(file_path)
        return file.get_structure(template)
    elif suffix == ".xtc":
        if template is None:
            raise TypeError("Template must be specified for trajectory files")
        from .xtc import XTCFile
        file = XTCFile()
        file.read(file_path)
        return file.get_structure(template)
    elif suffix == ".tng":
        if template is None:
            raise TypeError("Template must be specified for trajectory files")
        from .tng import TNGFile
        file = TNGFile()
        file.read(file_path)
        return file.get_structure(template)
    else:
        raise ValueError("Unknown file format")

