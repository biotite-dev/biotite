.. include:: /tutorial/preamble.rst

Measuring geometric quantities
==============================

.. currentmodule:: biotite.structure

A central functionality of the :mod:`biotite.structure` subpackage are its
efficient structure analysis capabilities, reaching from simple bond angle and
length measurements to more complex characteristics, like accessible surface
area and secondary structure.
This chapter will outline the provided toolset.

Distances, angles and dihedrals
-------------------------------
Let's start by measuring the distance between atoms.
The :func:`distance()` function is quite flexible:
We are able to pick any combination of an :class:`Atom`, :class:`AtomArray`
or :class:`AtomArrayStack` and we can even provide coordinates directly.
The input values are
`broadcasted <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_
as we would expect it from a *NumPy* :class:`ndarray`.

.. jupyter-execute::

    from tempfile import gettempdir
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx
    import biotite.database.rcsb as rcsb

    pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch("1l2y", "bcif", gettempdir()))
    stack = pdbx.get_structure(pdbx_file)
    # Filter only CA atoms
    stack = stack[:, stack.atom_name == "CA"]

    # Calculate distance between first and second CA in first frame
    array = stack[0]
    print("Atom to atom:", struc.distance(array[0], array[1]))
    # Calculate distance between the first atom
    # and all other CA atoms in the array
    print("Array to atom:")
    print(struc.distance(array[0], array))
    # Calculate pairwise distances between the CA atoms in the first frame
    # and the CA atoms in the second frame
    print("Array to array:")
    print(struc.distance(stack[0], stack[1]))
    # Calculate the distances between all CA atoms in the stack
    # and the first CA atom in the first frame
    # The resulting array is too large, therefore only the shape is printed
    print("Stack to atom:")
    print(struc.distance(stack, stack[0,0]).shape)
    # Calculate distances between all adjacent CA in the first frame
    print("Adjacent CA distances:")
    print(struc.distance(array[:-1], array[1:]))
    # Coordinates can be supplied directly
    print(
        "Distances between coordinates:",
        struc.distance(np.array([0,0,0]), np.array([1,1,1]))
    )

Note that both the given atoms/coordinates do not need to be from the same
structure.
You can also perform measurements between atoms from different structures.
Probably you want to superimpose the structures before that, as explained in
the :doc:`previous chapter <manipulation>`.

The functions :func:`angle()` and :func:`dihedral()` work analogously.

.. jupyter-execute::

    # Calculate angle between first 3 CA atoms in first frame
    # (in radians)
    print("Angle:", struc.angle(array[0],array[1],array[2]))
    # Calculate dihedral angle between first 4 CA atoms in first frame
    # (in radians)
    print("Dihedral angle:", struc.dihedral(array[0],array[1],array[2],array[4]))

Backbone dihedral angles
^^^^^^^^^^^^^^^^^^^^^^^^
Specifically for proteins, :func:`dihedral_backbone` measures the dihedral
angles of the peptide backbone: :math:`\phi`, :math:`\psi` and :math:`\omega`.
In the following code snippet we measure these angles and create a
simple *Ramachandran* plot for the first frame of *TC5b*.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import numpy as np
    from biotite import colors

    array = pdbx.get_structure(pdbx_file, model=1)
    phi, psi, omega = struc.dihedral_backbone(array)

    fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)
    ax.plot(
        phi * 360/(2*np.pi), psi * 360/(2*np.pi),
        marker="o", linestyle="None", color=colors["dimorange"]
    )
    ax.set_xlim(-180,180)
    ax.set_ylim(-180,180)
    ax.set_xlabel(r"$\phi$")
    _ = ax.set_ylabel(r"$\psi$")

Surface area
------------
Often another quantity of interest is the *solvent accessible surface area*
(SASA) that indicates whether an atom or residue is on the surface of the
structure or buried inside.
The function :func:`sasa()` approximates the SASA for each atom.
Then we can sum up the values for each residue, to get the residue-wise SASA.

Besides other parameters, you can choose between different
Van-der-Waals radii sets:
*ProtOr*, the default set, is a set that defines radii for
non-hydrogen atoms, but determines the radius of an atom based on the
assumed amount of hydrogen atoms connected to it.
Therefore, ``"ProtOr"`` is suitable for structures with missing hydrogen
atoms, like crystal structures.
Otherwise if hydrogen atoms are resolved, using ``"Single"`` is more
accurate, as a radius is assigned to every single atom.

.. jupyter-execute::

    array = pdbx.get_structure(pdbx_file, model=1)
    # The following line calculates the atom-wise SASA of the atom array
    atom_sasa = struc.sasa(array, vdw_radii="Single")
    # Sum up SASA for each residue in atom array
    res_sasa = struc.apply_residue_wise(array, atom_sasa, np.sum)

    # Now assume hydrogen atoms are not resolved
    array = array[array.element != "H"]
    approx_atom_sasa = struc.sasa(array, vdw_radii="ProtOr")
    approx_res_sasa = struc.apply_residue_wise(array, approx_atom_sasa, np.sum)

    fig, ax = plt.subplots(figsize=(6.0, 2.0), constrained_layout=True)
    labels = np.arange(1,21)
    ax.plot(labels, res_sasa, label="all atom", color=colors["dimorange"])
    ax.plot(labels, approx_res_sasa, label="approx.", color=colors["darkgreen"])
    ax.set_xlim(0,21)
    ax.set_xticks(labels)
    ax.set_xlabel("Residue")
    ax.set_ylabel("SASA (Å)")
    _ = ax.legend()

Secondary structure
-------------------

*Biotite* can also be used to assign *secondary structure elements* (SSE) to a
structure with the :func:`annotate_sse()` function.
An ``'a'`` means alpha-helix, ``'b'`` beta-sheet, and ``'c'`` means coil.

.. jupyter-execute::

    array = pdbx.get_structure(pdbx_file, model=1)
    array = array[array.chain_id == 'A']
    # Estimate secondary structure
    sse = struc.annotate_sse(array)
    # Pretty print
    print("".join(sse))

Note that you can also use the popular *DSSP* program to measure the secondary
structure as explained in a :doc:`later chapter <application/dssp>`.
