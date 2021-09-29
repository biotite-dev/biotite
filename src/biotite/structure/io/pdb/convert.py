# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Some convenience functions for consistency with other ``structure.io``
subpackages.
"""

__name__ = "biotite.structure.io.pdb"
__author__ = "Patrick Kunzmann"
__all__ = ["get_model_count", "get_structure", "set_structure"]


def get_model_count(pdb_file):
    """
    Get the number of models contained in a :class:`PDBFile`.

    Parameters
    ----------
    pdb_file : PDBFile
        The file object.

    Returns
    -------
    model_count : int
        The number of models.
    """
    return pdb_file.get_model_count()


def get_structure(pdb_file, model=None, altloc="first", extra_fields=[],
                  include_bonds=False):
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
        model number (starting at 1).
        Negative values are used to index models starting from the last
        model insted of the first model.
        If this parameter is omitted, an :class:`AtomArrayStack`
        containing all models will be returned, even if the structure
        contains only one model.
    altloc : {'first', 'occupancy', 'all'}
        This parameter defines how *altloc* IDs are handled:
            - ``'first'`` - Use atoms that have the first *altloc* ID
              appearing in a residue.
            - ``'occupancy'`` - Use atoms that have the *altloc* ID
              with the highest occupancy for a residue.
            - ``'all'`` - Use all atoms.
              Note that this leads to duplicate atoms.
              When this option is chosen, the ``altloc_id`` annotation
              array is added to the returned structure.
    extra_fields : list of str, optional
        The strings in the list are optional annotation categories
        that should be stored in the output array or stack.
        These are valid values:
        ``'atom_id'``, ``'b_factor'``, ``'occupancy'`` and ``'charge'``.
    include_bonds : bool, optional
        If set to true, a :class:`BondList` will be created for the
        resulting :class:`AtomArray` containing the bond information
        from the file.
        All bonds have :attr:`BondType.ANY`, since the PDB format
        does not support bond orders.
        
    Returns
    -------
    array : AtomArray or AtomArrayStack
        The return type depends on the `model` parameter.
    
    """
    return pdb_file.get_structure(model, altloc, extra_fields, include_bonds)


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

    Notes
    -----
    If `array` has an associated :class:`BondList`, ``CONECT``
    records are also written for all non-water hetero residues
    and all inter-residue connections.
    """
    pdb_file.set_structure(array, hybrid36)