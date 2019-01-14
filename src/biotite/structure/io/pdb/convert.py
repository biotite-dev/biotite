# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Some convenience functions for consistency with other ``structure.io``
subpackages.
"""

__author__ = "Patrick Kunzmann"
__all__ = ["get_structure", "set_structure"]


def get_structure(pdb_file, model=None,
                  insertion_code=[], altloc=[], extra_fields=[]):
    """
    Create an `AtomArray` or `AtomArrayStack` from a `PDBFile`.

    This function is a thin wrapper around the `PDBFile` method
    `get_structure()` for the sake of consistency with other
    ``structure.io`` subpackages.
    
    Parameters
    ----------
    pdb_file : PDBFile
        The file object.
    model : int, optional
        If this parameter is given, the function will return an
        `AtomArray` from the atoms corresponding to the given model ID.
        If this parameter is omitted, an `AtomArrayStack` containing all
        models will be returned, even if the structure contains only one
        model.
    insertion_code : list of tuple, optional
        In case the structure contains insertion codes, those can be
        specified here: Each tuple consists of an integer, specifying
        the residue ID, and a letter, specifying the insertion code.
        By default no insertions are used.
    altloc : list of tuple, optional
        In case the structure contains *altloc* entries, those can be
        specified here: Each tuple consists of an integer, specifying
        the residue ID, and a letter, specifying the *altloc* ID.
        By default the location with the *altloc* ID "A" is used.
    extra_fields : list of str, optional
        The strings in the list are entry names, that are
        additionally added as annotation arrays.
        The annotation category name will be the same as the PDBx
        subcategroy name. The array type is always `str`.
        There are 4 special field identifiers:
        'atom_id', 'b_factor', 'occupancy' and 'charge'.
        These will convert the respective subcategory into an
        annotation array with reasonable type.
        
    Returns
    -------
    array : AtomArray or AtomArrayStack
        The return type depends on the `model` parameter.
    
    """
    return pdb_file.get_structure(model, insertion_code, altloc, extra_fields)


def set_structure(pdb_file, array):
    """
    write an `AtomArray` or `AtomArrayStack` into a `PDBFile`.

    This function is a thin wrapper around the `PDBFile` method
    `set_structure()` for the sake of consistency with other
    ``structure.io`` subpackages.
    
    This will save the coordinates, the mandatory annotation categories
    and the optional annotation categories
    'atom_id', 'b_factor', 'occupancy' and 'charge'.
    
    Parameters
    ----------
    pdb_file : PDBFile
        The file object.
    array : AtomArray or AtomArrayStack
        The structure to be written. If a stack is given, each array in
        the stack will be in a separate model.
    """
    pdb_file.set_structure(array)