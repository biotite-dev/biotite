.. include:: /tutorial/preamble.rst

Representing bonds
==================

.. currentmodule:: biotite.structure

Up to now we only looked into atom arrays whose atoms are merely described by
its coordinates and annotations.
But there is more:
Chemical bonds can be described, too, using a :class:`BondList`.

Consider the following case, where our :class:`AtomArray` contains four atoms:
``N``, ``CA``, ``C`` and ``CB``. ``CA`` is a central atom that is connected to
``N``, ``C`` and ``CB``.
A :class:`BondList` is created by passing a :class:`ndarray` containing pairs
of integers, where each integer represents an index in a corresponding
:class:`AtomArray`.
The pairs indicate which atoms share a bond.
Additionally, it is required to specify the number of atoms in the
:class:`AtomArray`.

.. jupyter-execute::

    import biotite.structure as struc

    array = struc.array([
        struc.Atom([0,0,0], atom_name="N"),
        struc.Atom([0,0,0], atom_name="CA"),
        struc.Atom([0,0,0], atom_name="C"),
        struc.Atom([0,0,0], atom_name="CB")
    ])
    print("Atoms:", array.atom_name)
    bond_list = struc.BondList(
        array.array_length(),
        np.array([[1,0], [1,2], [1,3]])
    )
    print("Bonds (indices and type):")
    print(bond_list.as_array())
    print("Bonds (atoms names):")
    print(array.atom_name[bond_list.as_array()[:, :2]])
    ca_bonds, ca_bond_types = bond_list.get_bonds(1)
    print("Bonds of CA:", array.atom_name[ca_bonds])


When you look at the internal :class:`ndarray` (as given by
:func:`BondList.as_array()`), you see a third column containing zeros.
This column describes each bond with values from the :class:`BondType` enum:
``0`` corresponds to ``BondType.ANY``, which means that the type of the bond
is undefined.
This makes sense, since we did not define the bond types, when we created the
:class:`BondList`.
The other thing that has changed is the index order:
Each bond is sorted so that the index with the lower index is the
first element.

Although a :class:`BondList` uses a :class:`ndarray` under the hood, indexing
works a little bit different:
The indexing operation is not applied to the internal :class:`ndarray`, instead
it behaves like the same indexing operation was applied to a corresponding atom
array:
The bond list adjusts its indices so that they still point to the same atoms as
before.
Bonds that involve at least one atom, that has been removed, are
deleted as well.
We will try that by deleting the ``C`` atom.

.. jupyter-execute::

    mask = (array.atom_name != "C")
    sub_array = array[mask]
    sub_bond_list = bond_list[mask]
    print("Atoms:", sub_array.atom_name)
    print("Bonds (indices and type):")
    print(sub_bond_list.as_array())
    print("Bonds (atoms names):")
    print(sub_array.atom_name[sub_bond_list.as_array()[:, :2]])

As you see, the bond involving the ``C`` atom is removed and the remaining
indices are shifted.

Connecting atoms and bonds
--------------------------
We do not need to index the atom array and the bond list separately.
For the sake of convenience you can associate a :class:`BondList` to an
:class:`AtomArray` via the ``bonds`` attribute.
If no :class:`BondList` is associated, ``bonds`` is ``None``.
Every time the atom array is indexed, the index is also applied to the
associated bond list.

.. jupyter-execute::

    array.bonds = bond_list
    sub_array = array[array.atom_name != "C"]
    print("Bonds (atoms names):")
    print(sub_array.atom_name[sub_array.bonds.as_array()[:, :2]])

Keep in mind, that some functionalities in *Biotite* even require that the
input :class:`AtomArray` or :class:`AtomArrayStack` has an associated
:class:`BondList`.

Reading and writing bonds
-------------------------
Up to now the bond information has been created manually, which is impractical
in most cases.
Instead bond information can be loaded from and saved to most file formats.
We'll try that on the structure of *TC5b* and look at the bond information of
the third residue, a tyrosine.

.. jupyter-execute::

    from tempfile import gettempdir
    import biotite.database.rcsb as rcsb
    import biotite.structure.io.pdbx as pdbx

    file_path = rcsb.fetch("1l2y", "bcif", gettempdir())
    pdbx_file = pdbx.BinaryCIFFile.read(file_path)
    # Essential: set the 'include_bonds' parameter to true
    stack = pdbx.get_structure(pdbx_file, include_bonds=True)
    tyrosine = stack[:, (stack.res_id == 3)]
    print("Bonds (indices and type):")
    print(tyrosine.bonds.as_array())
    print("Bonds (atoms names):")
    print(tyrosine.atom_name[tyrosine.bonds.as_array()[:, :2]])

Not only the connected atoms, but also the bond types are defined:
Here we have both, ``BondType.SINGLE`` and ``BondType.DOUBLE`` bonds
(enum values ``1`` and ``2``, respectively).

Bond information can also be automatically inferred from an :class:`AtomArray`
or :class:`AtomArrayStack`:
:func:`connect_via_residue_names()` is able to connect atoms in all residues
that appear in the
`Chemical Component Dictionary <https://www.wwpdb.org/data/ccd>`_, comprising
every molecule that appears in any PDB entry.

.. jupyter-execute::

    stack = pdbx.get_structure(pdbx_file, include_bonds=False)
    stack.bonds = struc.connect_via_residue_names(stack)
    tyrosine = stack[:, (stack.res_id == 3)]
    print("Bonds (indices):")
    print(tyrosine.bonds.as_array())
    print("Bonds (atoms names):")
    print(tyrosine.atom_name[tyrosine.bonds.as_array()[:, :2]])

Filtering and editing bonds
---------------------------
The recommended way to apply changes to a :class:`BondList` (apart from adding/removing
single bonds) is to use the :class:`ndarray` obtained via :meth:`BondList.as_array()`
as transient representation and creating a new :class:`BondList` from the modified
:class:`ndarray`.

.. jupyter-execute::

    # Transiently convert the bond list to an array
    bond_array = tyrosine.bonds.as_array()
    # As an example, remove all single bonds
    bond_array = bond_array[bond_array[:, 2] != struc.BondType.SINGLE]
    # Create a new bond list from the modified array
    tyrosine.bonds = struc.BondList(tyrosine.array_length(), bond_array)
    print(tyrosine.atom_name[tyrosine.bonds.as_array()[:, :2]])