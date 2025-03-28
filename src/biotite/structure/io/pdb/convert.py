# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Some convenience functions for consistency with other ``structure.io``
subpackages.
"""

__name__ = "biotite.structure.io.pdb"
__author__ = "Patrick Kunzmann"
__all__ = [
    "get_model_count",
    "get_structure",
    "set_structure",
    "list_assemblies",
    "get_assembly",
    "get_unit_cell",
]

import warnings


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


def get_structure(
    pdb_file, model=None, altloc="first", extra_fields=[], include_bonds=False
):
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
        model instead of the first model.
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
        Bonds, whose order could not be determined from the
        *Chemical Component Dictionary*
        (e.g. especially inter-residue bonds),
        have :attr:`BondType.ANY`, since the PDB format itself does
        not support bond orders.

    Returns
    -------
    array : AtomArray or AtomArrayStack
        The return type depends on the `model` parameter.
    """
    return pdb_file.get_structure(model, altloc, extra_fields, include_bonds)


def set_structure(pdb_file, array, hybrid36=False):
    """
    Write an :class:`AtomArray` or :class:`AtomArrayStack` into a
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
    hybrid36 : boolean, optional
        Defines wether the file should be written in hybrid-36 format.

    Notes
    -----
    If `array` has an associated :class:`BondList`, ``CONECT``
    records are also written for all non-water hetero residues
    and all inter-residue connections.
    """
    pdb_file.set_structure(array, hybrid36)


def list_assemblies(pdb_file):
    """
    List the biological assemblies that are available for the
    structure in the given file.

    This function receives the data from the ``REMARK 300`` records
    in the file.
    Consequently, this remark must be present in the file.

    Parameters
    ----------
    pdb_file : PDBFile
        The file object.

    Returns
    -------
    assemblies : list of str
        A list that contains the available assembly IDs.

    Examples
    --------
    >>> import os.path
    >>> file = PDBFile.read(os.path.join(path_to_structures, "1f2n.pdb"))
    >>> print(list_assemblies(file))
    ['1']
    """
    return pdb_file.list_assemblies()


def get_assembly(
    pdb_file,
    assembly_id=None,
    model=None,
    altloc="first",
    extra_fields=[],
    include_bonds=False,
):
    """
    Build the given biological assembly.

    This function receives the data from ``REMARK 350`` records in
    the file.
    Consequently, this remark must be present in the file.

    Parameters
    ----------
    pdb_file : PDBFile
        The file object.
    assembly_id : str
        The assembly to build.
        Available assembly IDs can be obtained via
        :func:`list_assemblies()`.
    model : int, optional
        If this parameter is given, the function will return an
        :class:`AtomArray` from the atoms corresponding to the given
        model number (starting at 1).
        Negative values are used to index models starting from the
        last model instead of the first model.
        If this parameter is omitted, an :class:`AtomArrayStack`
        containing all models will be returned, even if the
        structure contains only one model.
    altloc : {'first', 'occupancy', 'all'}
        This parameter defines how *altloc* IDs are handled:
            - ``'first'`` - Use atoms that have the first
                *altloc* ID appearing in a residue.
            - ``'occupancy'`` - Use atoms that have the *altloc* ID
                with the highest occupancy for a residue.
            - ``'all'`` - Use all atoms.
                Note that this leads to duplicate atoms.
                When this option is chosen, the ``altloc_id``
                annotation array is added to the returned structure.
    extra_fields : list of str, optional
        The strings in the list are optional annotation categories
        that should be stored in the output array or stack.
        These are valid values:
        ``'atom_id'``, ``'b_factor'``, ``'occupancy'`` and
        ``'charge'``.
    include_bonds : bool, optional
        If set to true, a :class:`BondList` will be created for the
        resulting :class:`AtomArray` containing the bond information
        from the file.
        Bonds, whose order could not be determined from the
        *Chemical Component Dictionary*
        (e.g. especially inter-residue bonds),
        have :attr:`BondType.ANY`, since the PDB format itself does
        not support bond orders.

    Returns
    -------
    assembly : AtomArray or AtomArrayStack
        The assembly.
        The return type depends on the `model` parameter.
        Contains the `sym_id` annotation, which enumerates the copies of the asymmetric
        unit in the assembly.

    Examples
    --------

    >>> import os.path
    >>> file = PDBFile.read(os.path.join(path_to_structures, "1f2n.pdb"))
    >>> assembly = get_assembly(file, model=1)
    """
    return pdb_file.get_assembly(
        assembly_id, model, altloc, extra_fields, include_bonds
    )


def get_unit_cell(
    pdb_file, model=None, altloc="first", extra_fields=[], include_bonds=False
):
    """
    Build a structure model containing all symmetric copies
    of the structure within a single unit cell, given by the space
    group.

    This function receives the data from ``REMARK 290`` records in
    the file.
    Consequently, this remark must be present in the file, which is
    usually only true for crystal structures.

    Parameters
    ----------
    pdb_file : PDBFile
        The file object.
    model : int, optional
        If this parameter is given, the function will return an
        :class:`AtomArray` from the atoms corresponding to the given
        model number (starting at 1).
        Negative values are used to index models starting from the
        last model instead of the first model.
        If this parameter is omitted, an :class:`AtomArrayStack`
        containing all models will be returned, even if the
        structure contains only one model.
    altloc : {'first', 'occupancy', 'all'}
        This parameter defines how *altloc* IDs are handled:
            - ``'first'`` - Use atoms that have the first
                *altloc* ID appearing in a residue.
            - ``'occupancy'`` - Use atoms that have the *altloc* ID
                with the highest occupancy for a residue.
            - ``'all'`` - Use all atoms.
                Note that this leads to duplicate atoms.
                When this option is chosen, the ``altloc_id``
                annotation array is added to the returned structure.
    extra_fields : list of str, optional
        The strings in the list are optional annotation categories
        that should be stored in the output array or stack.
        These are valid values:
        ``'atom_id'``, ``'b_factor'``, ``'occupancy'`` and
        ``'charge'``.
    include_bonds : bool, optional
        If set to true, a :class:`BondList` will be created for the
        resulting :class:`AtomArray` containing the bond information
        from the file.
        Bonds, whose order could not be determined from the
        *Chemical Component Dictionary*
        (e.g. especially inter-residue bonds),
        have :attr:`BondType.ANY`, since the PDB format itself does
        not support bond orders.

    Returns
    -------
    symmetry_mates : AtomArray or AtomArrayStack
        All atoms within a single unit cell.
        The return type depends on the `model` parameter.

    Notes
    -----
    To expand the structure beyond a single unit cell, use
    :func:`repeat_box()` with the return value as its
    input.

    Examples
    --------

    >>> import os.path
    >>> file = PDBFile.read(os.path.join(path_to_structures, "1aki.pdb"))
    >>> atoms_in_unit_cell = get_unit_cell(file, model=1)
    """
    return pdb_file.get_unit_cell(model, altloc, extra_fields, include_bonds)


def get_symmetry_mates(
    pdb_file, model=None, altloc="first", extra_fields=[], include_bonds=False
):
    """
    Build a structure model containing all symmetric copies
    of the structure within a single unit cell, given by the space
    group.

    This function receives the data from ``REMARK 290`` records in
    the file.
    Consequently, this remark must be present in the file, which is
    usually only true for crystal structures.

    DEPRECATED: Use :func:`get_unit_cell()` instead.

    Parameters
    ----------
    pdb_file : PDBFile
        The file object.
    model : int, optional
        If this parameter is given, the function will return an
        :class:`AtomArray` from the atoms corresponding to the given
        model number (starting at 1).
        Negative values are used to index models starting from the
        last model instead of the first model.
        If this parameter is omitted, an :class:`AtomArrayStack`
        containing all models will be returned, even if the
        structure contains only one model.
    altloc : {'first', 'occupancy', 'all'}
        This parameter defines how *altloc* IDs are handled:
            - ``'first'`` - Use atoms that have the first
                *altloc* ID appearing in a residue.
            - ``'occupancy'`` - Use atoms that have the *altloc* ID
                with the highest occupancy for a residue.
            - ``'all'`` - Use all atoms.
                Note that this leads to duplicate atoms.
                When this option is chosen, the ``altloc_id``
                annotation array is added to the returned structure.
    extra_fields : list of str, optional
        The strings in the list are optional annotation categories
        that should be stored in the output array or stack.
        These are valid values:
        ``'atom_id'``, ``'b_factor'``, ``'occupancy'`` and
        ``'charge'``.
    include_bonds : bool, optional
        If set to true, a :class:`BondList` will be created for the
        resulting :class:`AtomArray` containing the bond information
        from the file.
        Bonds, whose order could not be determined from the
        *Chemical Component Dictionary*
        (e.g. especially inter-residue bonds),
        have :attr:`BondType.ANY`, since the PDB format itself does
        not support bond orders.

    Returns
    -------
    symmetry_mates : AtomArray or AtomArrayStack
        All atoms within a single unit cell.
        The return type depends on the `model` parameter.

    Notes
    -----
    To expand the structure beyond a single unit cell, use
    :func:`repeat_box()` with the return value as its
    input.

    Examples
    --------

    >>> import os.path
    >>> file = PDBFile.read(os.path.join(path_to_structures, "1aki.pdb"))
    >>> atoms_in_unit_cell = get_symmetry_mates(file, model=1)
    """
    warnings.warn(
        "'get_symmetry_mates()' is deprecated, use 'get_unit_cell()' instead",
        DeprecationWarning,
    )
    return pdb_file.get_unit_cell(model, altloc, extra_fields, include_bonds)
