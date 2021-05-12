# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Some convenience functions for consistency with other ``structure.io``
subpackages.
"""

__name__ = "biotite.structure.io.pdbqt"
__author__ = "Patrick Kunzmann"
__all__ = ["get_structure", "set_structure"]


def get_structure(pdbqt_file, model=None):
    """
    Get an :class:`AtomArray` or :class:`AtomArrayStack` from the
    PDBQT file.

    EXPERIMENTAL: Future API changes are probable.
    
    Parameters
    ----------
    pdbqt_file : PDBQTFile
        The PDBQT file.
    model : int, optional
        If this parameter is given, the function will return an
        :class:`AtomArray` from the atoms corresponding to the given
        model number (starting at 1).
        Negative values are used to index models starting from the
        last model insted of the first model.
        If this parameter is omitted, an :class:`AtomArrayStack`
        containing all models will be returned, even if the
        structure contains only one model.
    
    Returns
    -------
    array : AtomArray or AtomArrayStack
        The return type depends on the `model` parameter.
    """
    return pdbqt_file.get_structure(model)


def set_structure(pdbqt_file, atoms, charges=None, atom_types=None,
                  rotatable_bonds=None, root=None, include_torsdof=True):
    """
    Write an :class:`AtomArray` into a PDBQT file.

    EXPERIMENTAL: Future API changes are probable.
    
    Parameters
    ----------
    pdbqt_file : PDBQTFile
        The PDBQT file.
    atoms : AtomArray, shape=(n,)
        The atoms to be written into this file.
        Must have an associated :class:`BondList`.
    charges : ndarray, shape=(n,), dtype=float, optional
        Partial charges for each atom in `atoms`.
        By default, the charges are calculated using the PEOE method
        (:func:`partial_charges()`).
    atom_types : ndarray, shape=(n,), dtype="U1", optional
        Custom *AutoDock* atom types for each atom in `atoms`.
    rotatable_bonds : None or 'rigid' or 'all' or BondList, optional
        This parameter describes, how rotatable bonds are handled,
        with respect to ``ROOT``, ``BRANCH`` and ``ENDBRANCH``
        lines.

            - ``None`` - The molecule is handled as rigid receptor:
              No ``ROOT``, ``BRANCH`` and ``ENDBRANCH`` lines will
              be written.
            - ``'rigid'`` - The molecule is handled as rigid ligand:
              Only a ``ROOT`` line will be written.
            - ``'all'`` - The molecule is handled as flexible 
              ligand:
              A ``ROOT`` line will be written and all rotatable
              bonds are included using ``BRANCH`` and ``ENDBRANCH``
              lines.
            - :class:`BondList` - The molecule is handled as
              flexible ligand:
              A ``ROOT`` line will be written and all bonds in the
              given :class:`BondList` are considered flexible via
              ``BRANCH`` and ``ENDBRANCH`` lines.
        
    root : int, optional
        Specifies the index of the atom following the ``ROOT`` line.
        Setting the root atom is useful for specifying the *anchor*
        in flexible side chains.
        This parameter has no effect, if `rotatable_bonds` is
        ``None``.
        By default, the first atom is also the root atom.
    include_torsdof : bool, optional
        By default, a ``TORSDOF`` (torsional degrees of freedom)
        record is written at the end of the file.
        By setting this parameter to false, the record is omitted.
    
    Returns
    -------
    mask : ndarray, shape=(n,), dtype=bool
        A boolean mask, that is ``False`` for each atom of the input
        ``atoms``, that was removed due to being a nonpolar
        hydrogen.
    """
    return pdbqt_file.set_structure(
        atoms, charges, atom_types, rotatable_bonds, root,
        include_torsdof
    )