# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.xyz"
__author__ = "Benjamin E. Mayer"
__all__ = ["get_structure", "set_structure"]



def get_structure(xyz_file, model=None):
    """
    Get an :class:`AtomArray` from the XYZ File.

    Ths function is a thin wrapper around
    :meth:`XYZFile.get_structure()`.

    Parameters
    ----------
    xyz_file : XYZFile
        The XYZ File.
    
    Returns
    -------
    array : AtomArray
        This :class:`AtomArray` contains the optional ``charge``
        annotation and has an associated :class:`BondList`.
        All other annotation categories, except ``element`` are
        empty.
    """
    return xyz_file.get_structure(model)
    

def set_structure(xyz_file, atoms):
    """
    Set the :class:`AtomArray` for the XYZ File
    or :class:`AtomArrayStack for a XYZ File with multiple states.

    Ths function is a thin wrapper around
    :meth:`XYZFile.set_structure()`.
    
    Parameters
    ----------
    xyz_file : XYZFile
        The XYZ File.
    array : AtomArray or AtomArrayStack
        The array to be saved into this file. If a stack is given, each array in
        the stack will be in a separate model and the names of the 
        model will be set to an index enumerating them.
    """
    xyz_file.set_structure(atoms)
