.. include:: /tutorial/preamble.rst

Array indexing and filtering
============================

.. currentmodule:: biotite.structure

:class:`AtomArray` and :class:`AtomArrayStack` objects can be indexed in a
similar way a :class:`ndarray` is indexed.
In fact, the index is propagated to the coordinates and the annotation arrays.
Therefore, all *NumPy* compatible types of indices can be used, like boolean
arrays, index arrays/lists, slices and, of course, integer values.
Integer indices have a special role here, as they reduce the dimensionality of
the data type:
Indexing an :class:`AtomArrayStack` with an integer results in an
:class:`AtomArray` at the specified frame, indexing an :class:`AtomArray` with
an integer yields the specified :class:`Atom`.
Iterating over arrays and stacks reduces the dimensionality in an analogous
way.
Let's demonstrate indexing with the help of the structure of *TC5b*.

.. jupyter-execute::

    from tempfile import gettempdir
    import biotite.structure as struc
    import biotite.database.rcsb as rcsb
    import biotite.structure.io.pdbx as pdbx

    pdbx_file = pdbx.BinaryCIFFile.read(
        rcsb.fetch("1l2y", "bcif", gettempdir())
    )
    stack = pdbx.get_structure(pdbx_file)
    print(type(stack).__name__)
    print(stack.shape)
    # Get the third model
    array = stack[2]
    print(type(array).__name__)
    print(array.shape)
    # Get the fifth atom
    atom = array[4]
    print(type(atom).__name__)
    print(atom.shape)

:func:`get_structure()` gives us an :class:`AtomArrayStack`.
The first indexing step reduces the stack to an atom array and the second
indexing step reduces the array to a single atom.
The `shape` attribute gives the number of models and atoms, similarly to the
`shape` attribute of :class:`ndarray` objects.
Alternatively, the :func:`stack_depth()` or :func:`array_length()` methods can
be used to get the number of models or atoms, respectively.

The following code section shows some examples for how an atom array can be
indexed.

.. jupyter-execute::

    # Get the first atom
    atom = array[0]
    # Get a subarray containing the first and third atom
    subarray = array[[0,2]]
    # Get a subarray containing a range of atoms using slices
    subarray = array[100:200]
    # Filter all carbon atoms in residue 1
    subarray = array[(array.element == "C") & (array.res_id == 1)]
    # Filter all atoms where the X-coordinate is smaller than 2
    subarray = array[array.coord[:,0] < 2]

An atom array stack can be indexed in a similar way, with the difference, that
the index specifies the frame(s).

.. jupyter-execute::

    # Get an atom array from the first model
    subarray = stack[0]
    # Get a substack containing the first 10 models
    substack = stack[:10]

Stacks also have the speciality, that they can handle 2-dimensional indices,
where the first dimension specifies the frame(s) and the second dimension
specifies the atom(s).

.. jupyter-execute::

    # Get the first 100 atoms from the third model
    subarray = stack[2, :100]
    # Get the first 100 atoms from the models 3, 4 and 5
    substack = stack[2:5, :100]
    # Get the first atom in the second model
    atom = stack[1,0]
    # Get a stack containing arrays containing only the first atom
    substack = stack[:, 0]

Furthermore, :mod:`biotite.structure` contains advanced filters,
that create boolean masks from an atom array using specific criteria.
Here is a small example.

.. jupyter-execute::

    backbone = array[struc.filter_peptide_backbone(array)]
    print(backbone.atom_name)
