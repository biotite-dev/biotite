.. include:: /tutorial/preamble.rst

From single atoms to multi-model structures
===========================================

.. currentmodule:: biotite.structure

To understand how :class:`Atom`, :class:`AtomArray` and :class:`AtomArrayStack`
relate to each other, we will create them from scratch.
In an actual application one would usually read a structure from a file, as
explained :doc:`in the next chapter <io>`.

.. jupyter-execute::

    import biotite.structure as struc

    atom1 = struc.Atom(
        [0,0,0], chain_id="A", res_id=1, res_name="GLY",
        atom_name="N", element="N"
    )
    atom2 = struc.Atom(
        [0,1,1], chain_id="A", res_id=1, res_name="GLY",
        atom_name="CA", element="C"
    )
    atom3 = struc.Atom(
        [0,0,2], chain_id="A", res_id=1, res_name="GLY",
        atom_name="C", element="C"
    )

The first parameter is the coordinates (internally converted into an
:class:`ndarray`), the other parameters are annotations.
The annotations shown in this example are mandatory:
The chain ID, residue ID, residue name, insertion code, atom name, element and
whether the atom is not in protein/nucleotide chain (*hetero*).
If you miss one of these, they will get a default value.
The mandatory annotation categories originate from the ``ATOM`` and ``HETATM``
records in the PDB format.
Additionally, you can specify an arbitrary amount of custom annotations, like
B-factors, charge, etc.

In most cases you won't work with single :class:`Atom` instances, because one
usually deals with entire molecular structures, containing an arbitrary amount
of atoms.
For this purpose :mod:`biotite.structure` offers the :class:`AtomArray`.
An atom array can be seen as an array of atom instances (hence the name).
But instead of storing :class:`Atom` instances in a list, an :class:`AtomArray`
instance contains one :class:`ndarray` for each annotation and the coordinates.
In order to see this in action, we first have to create an array from
the atoms we constructed before.
Then we can access the annotations and coordinates of the atom array simply by
specifying the attribute.

.. jupyter-execute::

    import numpy as np

    array = struc.array([atom1, atom2, atom3])
    print("Chain ID:", array.chain_id)
    print("Residue ID:", array.res_id)
    print("Atom name:", array.atom_name)
    print("Coordinates:", array.coord)
    print()
    print(array)

The :func:`array()` builder function takes any iterable object containing
:class:`Atom` instances.
If you wanted to, you could even use another :class:`AtomArray`, which
functions also as an iterable object of :class:`Atom` objects.
An alternative way of constructing an array would be creating an
:class:`AtomArray` by using its constructor, which fills the annotation arrays
and coordinates with the type-specific *zero* value.
In our example all annotation arrays have a length of 3, since we used 3 atoms
to create it.
A structure containing *n* atoms is represented by annotation arrays of length
*n* and coordinates of shape *(n,3)*.
As the annotations and coordinates are simply :class:`ndarray` objects, they
can be edited using *NumPy* functionality.

.. jupyter-execute::

    array.chain_id[:] = "B"
    array.coord[array.element == "C", 0] = 42
    # It is also possible to replace an entire annotation with another array
    array.res_id = np.array([1,2,3])
    print(array)

Apart from the structure manipulation functions we see later on, this
is the usual way to edit structures in *Biotite*.

.. warning::

    For editing an annotation, the index must be applied to the annotation and
    not to the :class:`AtomArray`.
    For example, you should write ``array.chain_id[...] = "B"`` instead of
    ``array[...].chain_id = "B"``.
    The latter example is incorrect, as it creates a subarray of the initial
    :class:`AtomArray` (discussed in a :doc:`later chapter <filter>`) and then
    tries to replace the annotation array with the new value.

If you want to add further annotation categories to an array, you have to call
the :func:`add_annotation()` or :func:`set_annotation()` method at first.
After that you can access the new annotation array like any other annotation
array.

.. jupyter-execute::

    array.add_annotation("foo", dtype=bool)
    array.set_annotation("bar", [1, 2, 3])
    print(array.foo)
    print(array.bar)

In some cases, you might need to handle structures, where each atom is
present in multiple locations
(multiple models in NMR structures, MD trajectories).
For these cases :class:`AtomArrayStack` objects enter the stage:
They represent a list of atom arrays with the same atoms in each model/frame,
but differing coordinates.
Hence the annotation arrays in :class:`AtomArrayStack` objects still have the
same length *n* as in :class:`AtomArray`.
However, a stack stores the coordinates in a *(m,n,3)*-shaped
:class:`ndarray`, where *m* is the number of frames.
A stack is constructed with :func:`stack()` analogous to the code
snippet above.
It is crucial that all :class:`AtomArray` objects, that should be stacked,
have the same annotation arrays, otherwise an exception is raised.
For simplicity reasons, we create a stack containing two identical
models, derived from the previous example.

.. jupyter-execute::

    stack = struc.stack([array, array.copy()])
    print(stack)