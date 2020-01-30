# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Some convenience functions for consistency with other ``structure.io``
subpackages.
"""

__name__ = "biotite.structure.io.pdb"
__author__ = "Patrick Kunzmann"
__all__ = ["get_structure", "set_structure"]


def get_structure(pdb_file, model=None, altloc=[], extra_fields=[]):
    """
    Create an :class:`AtomArray` or :class:`AtomArrayStack` from a
    :class:`PDBFile`.

    This function is a thin wrapper around the :class:`PDBFile` method
    :func:`get_structure()` for the sake of consistency with other
    ``structure.io`` subpackages.
    
    Parameters
    ----------
    pdb_file : PDBFile
        The file object.
    model : int, optional
        If this parameter is given, the function will return an
        :class:`AtomArray` from the atoms corresponding to the given
        model number.
        If this parameter is omitted, an :class:`AtomArrayStack`
        containing all models will be returned, even if the
        structure contains only one model.
    altloc : list of tuple, optional
        In case the structure contains *altloc* entries, those can be
        specified here:
        Each tuple consists of the following elements:

            - A chain ID, specifying the residue
            - A residue ID, specifying the residue
            - The desired *altoc* ID for the specified residue

        For each of the given residues the atoms with the given *altloc*
        ID are filtered.
        By default the location with the *altloc* ID "A" is used.
    extra_fields : list of str, optional
        The strings in the list are optional annotation categories
        that should be stored in the output array or stack.
        These are valid values:
        ``'atom_id'``, ``'b_factor'``, ``'occupancy'`` and ``'charge'``.
        
    Returns
    -------
    array : AtomArray or AtomArrayStack
        The return type depends on the `model` parameter.
    
    """
    return pdb_file.get_structure(model, altloc, extra_fields)


def set_structure(pdb_file, array, hybrid36=False):
    """
    write an :class:`AtomArray` or :class:`AtomArrayStack` into a
    :class:`PDBFile`.

    This function is a thin wrapper around the :class:`PDBFile` method
    :func:`set_structure()` for the sake of consistency with other
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
    hybrid36: boolean, optional
        Defines wether the file should be written in hybrid-36 format.
    """
    pdb_file.set_structure(array, hybrid36)