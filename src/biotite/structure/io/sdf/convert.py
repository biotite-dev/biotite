# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.sdf"
__author__ = "Benjamin E. Mayer"
__all__ = ["get_structure", "set_structure"]



def get_structure(sdf_file):
    """
    Get an :class:`AtomArray` from the SDF File.

    Ths function is a thin wrapper around
    :meth:`SDFFile.get_structure()`.

    Parameters
    ----------
    sdf_file : SDFFile
        The SDF File.
    
    Returns
    -------
    array : AtomArray
        This :class:`AtomArray` contains the optional ``charge``
        annotation and has an associated :class:`BondList`.
        All other annotation categories, except ``element`` are
        empty.
    """
    return sdf_file.get_structure()
    

def set_structure(sdf_file, atoms):
    """
    Set the :class:`AtomArray` for the SDF File.

    Ths function is a thin wrapper around
    :meth:`SDFFile.set_structure()`.
    
    Parameters
    ----------
    sdf_file : SDFFile
        The SDF File.
    array : AtomArray
        The array to be saved into this file.
        Must have an associated :class:`BondList`.
    """
    sdf_file.set_structure(atoms)
