# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mol2"
__author__ = "Benjamin E. Mayer"
__all__ = [
    "get_structure", "set_structure",
    "get_charges", "set_charges"
]



def get_structure(mol2_file):
    """
    Get an :class:`AtomArray` from the MOL2 File.

    Ths function is a thin wrapper around
    :meth:`MOL2File.get_structure()`.

    Parameters
    ----------
    mol2_file : MOL2File
        The MOL2File.
    
    Returns
    -------
    array : AtomArray, AtomArrayStack
        Return an AtomArray or AtomArrayStack containing the structure or
        structures depending on if file contains single or multiple models.        
        If something other then `NO_CHARGE` is set in the charge_type field
        of the according mol2 file, the AtomArray or AtomArrayStack will 
        contain the charge field.
    """
    return mol2_file.get_structure()
    

def set_structure(mol2_file, atoms):
    """
    Set the :class:`AtomArray` for the MOL2 File.

    Ths function is a thin wrapper around
    :meth:`MOL2File.set_structure(atoms)`.
    
    Parameters
    ----------
    mol2_file : MOL2File
        The XYZ File.
    array : AtomArray
        The array to be saved into this file.
        Must have an associated :class:`BondList`.
        If charge field set this is used for storage within the according
        MOL2 charge column.
    """
    mol2_file.set_structure(atoms)
    
    

def get_charges(mol2_file):
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
    return mol2_file.get_charges()    
    

def set_charges(mol2_file, charges):
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
    return mol2_file.set_charges(charges)    
