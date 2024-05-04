.. include:: /tutorial/preamble.rst

Editing structures
==================

.. currentmodule:: biotite.structure

The most basic way to manipulate a structure is to edit the annotation arrays
or coordinates directly.

.. jupyter-execute::

    from tempfile import gettempdir
    import biotite.database.rcsb as rcsb
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch("1l2y", "bcif", gettempdir()))
    structure = pdbx.get_structure(pdbx_file, model=1)
    print("Before:")
    print(structure[structure.res_id == 1])
    print()
    structure.coord += 100
    print("After:")
    print(structure[structure.res_id == 1])

*Biotite* provides also some transformation functions, for example
:func:`rotate()` for rotations about the *x*-, *y*- or *z*-axis.

.. jupyter-execute::

    structure = pdbx.get_structure(pdbx_file, model=1)
    print("Before:")
    print(structure[structure.res_id == 1])
    print()
    # Rotation about z-axis by 90 degrees
    structure = struc.rotate(structure, [0, 0, np.deg2rad(90)])
    print("After:")
    print(structure[structure.res_id == 1])

Structure superimposition
-------------------------
A common prerequisite to compare two structures is the superimposing them onto
each other.
This means translating and rotating one structure so that the
*root mean square deviation* (RMSD) between corresponding atoms in the two
structures is minimized.
To demonstrate this, we will use two models of the *TC5b* structure already
used in the previous chapters.

.. jupyter-execute::

    reference = pdbx.get_structure(pdbx_file, model=1)
    # Rotate reference to remove the superimposition originating from the file
    reference = struc.rotate(reference, [np.deg2rad(45), 0, 0])
    mobile = pdbx.get_structure(pdbx_file, model=2)
    print(f"RMSD before superimposition: {struc.rmsd(reference, mobile):.2f}")
    superimposed, transformation = struc.superimpose(reference, mobile)
    print(f"RMSD after superimposition: {struc.rmsd(reference, superimposed):.2f}")

.. note::

    It is required that both structures have the same number (and order) of
    atoms, as the algorithm requires that each atom corresponds to an atom in
    the other structure.

The returned :class:`AffineTransformation` object can be used later to
transform another structure in the same way the mobile structure was
transformed.

.. jupyter-execute::

    another_model = pdbx.get_structure(pdbx_file, model=3)
    print(f"RMSD before transformation: {struc.rmsd(mobile, another_model):.2f}")
    # Apply the same transformation that was applied on the mobile structure
    transformed = transformation.apply(another_model)
    print(f"RMSD after transformation: {struc.rmsd(superimposed, transformed):.2f}")

We can see that both RMSD values are equal:
As the same transformation was applied to ``mobile`` and ``another_model``,
the atom positions relative to each other did not change.