.. include:: /tutorial/preamble.rst

Loading structures from file
============================
Usually structures are not built from scratch, but they are read from a file.
For this tutorial, we will work on a protein structure as small as possible,
namely the miniprotein *TC5b* (PDB: ``1L2Y``).
The structure of this 20-residue protein (304 atoms) has been elucidated via
NMR.
Thus, the corresponding structure file consists of multiple (namely 38) models, each
showing another conformation.

Reading PDB files
-----------------

.. currentmodule:: biotite.structure.io.pdb

Probably one of the most popular structure file formats to date is the
*Protein Data Bank Exchange* (PDB) format.
At first we load the structure from a PDB file via the class
:class:`PDBFile` in the subpackage :mod:`biotite.structure.io.pdb`.

.. jupyter-execute::

    from tempfile import gettempdir, NamedTemporaryFile
    import biotite.structure.io.pdb as pdb
    import biotite.database.rcsb as rcsb

    pdb_file_path = rcsb.fetch("1l2y", "pdb", gettempdir())
    pdb_file = pdb.PDBFile.read(pdb_file_path)
    tc5b = pdb_file.get_structure()
    print(type(tc5b).__name__)
    print(tc5b.stack_depth())
    print(tc5b.array_length())
    print(tc5b.shape)

The method :func:`PDBFile.get_structure()` returns a :class:`AtomArrayStack`
unless the :obj:`model` parameter is specified, even if the file contains only
one model.
The following example shows how to write an atom array or stack back into a
PDB file:

.. jupyter-execute::

    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(tc5b)
    temp_file = NamedTemporaryFile(suffix=".pdb", delete=False)
    pdb_file.write(temp_file.name)
    temp_file.close()

Other information (authors, secondary structure, etc.) cannot be
easily from PDB files using :class:`PDBFile`.

Working with the PDBx format
----------------------------

.. currentmodule:: biotite.structure.io.pdbx

After all, the *PDB* format itself is deprecated now due to several
shortcomings and was replaced by the *Protein Data Bank Exchange* (PDBx)
format.
As PDBx has become the standard structure format, it is also the format with
the most comprehensive interface in *Biotite*.
Today, this format has two common encodings:
The original text-based *Crystallographic Information Framework* (CIF)
and the *BinaryCIF* format.
While the former is human-readable, the latter is more efficient in terms of
file size and parsing speed.
The :mod:`biotite.structure.io.pdbx` subpackage provides classes for
interacting with both formats, :class:`CIFFile` and :class:`BinaryCIFFile`,
respectively.
In the following section we will focus on :class:`CIFFile`, but
:class:`BinaryCIFFile` works analogous.

.. jupyter-execute::

    import biotite.structure.io.pdbx as pdbx

    cif_file_path = rcsb.fetch("1l2y", "cif", gettempdir())
    cif_file = pdbx.CIFFile.read(cif_file_path)

*PDBx* can be imagined as hierarchical dictionary, with several
levels:

  #. **File**: The entirety of the *PDBx* file.
  #. **Block**: The data for a single structure (e.g. `1L2Y`).
  #. **Category**: A coherent group of data
     (e.g. `atom_site` describes the atoms).
     Each column in the category must have the same length.
  #. **Column**: Contains values of a specific type
     (e.g `atom_site.Cartn_x` contains the *x* coordinates for each atom).
     Contains two *Data* instances, one for the actual data and one
     for a mask.
     In a lot of categories a column contains only a single value.
  #. **Data**: The actual data in form of a :class:`ndarray`.

Each level may contain multiple instances of the next lower level, e.g. a
category may contain multiple columns.
Each level is represented by a separate class, that can be used like a
dictionary.
For CIF files these are :class:`CIFFile`, :class:`CIFBlock`,
:class:`CIFCategory`, :class:`CIFColumn` and :class:`CIFData`.
Note that :class:`CIFColumn` is not treated like a dictionary, but
instead has a ``data`` and ``mask`` attribute.

.. jupyter-execute::

    block = cif_file["1L2Y"]
    category = block["audit_author"]
    column = category["name"]
    data = column.data
    print(data.array)

The data access can be cut short, especially if the contains a single block
and a certain data type is expected instead of strings.

.. jupyter-execute::

    category = cif_file.block["audit_author"]
    column = category["pdbx_ordinal"]
    print(column.as_array(int))

As already mentioned, many categories contain only a single value per column.
In this case it may be convenient to get only a single item instead of an
array.

.. jupyter-execute::

    for key, column in cif_file.block["citation"].items():
        print(f"{key:25}{column.as_item()}")

Note the ``?`` in the output.
It indicates that the value is masked as '*unknown*'.
That becomes clear when we look at the mask of that column.

.. jupyter-execute::

    mask = block["citation"]["book_publisher"].mask.array
    print(mask)
    print(pdbx.MaskValue(mask[0]))


For setting/adding blocks, categories etc. we simply assign values as we would
do with dictionaries.

.. jupyter-execute::

    category = pdbx.CIFCategory()
    category["number"] = pdbx.CIFColumn(pdbx.CIFData([1, 2]))
    category["person"] = pdbx.CIFColumn(pdbx.CIFData(["me", "you"]))
    category["greeting"] = pdbx.CIFColumn(pdbx.CIFData(["Hi!", "Hello!"]))
    block["greetings"] = category
    print(category.serialize())

For the sake of brevity it is also possible to omit :class:`CIFColumn` and
:class:`CIFData` and even pass columns directly at category creation.

.. jupyter-execute::

    category = pdbx.CIFCategory({
        # If the columns contain only a single value, no list is required
        "fruit": "apple",
        "color": "red",
        "taste": "delicious",
    })
    block["fruits"] = category
    print(category.serialize())

For :class:`BinaryCIFFile` the usage is analogous.

.. jupyter-execute::

    bcif_file_path = rcsb.fetch("1l2y", "bcif", gettempdir())
    bcif_file = pdbx.BinaryCIFFile.read(bcif_file_path)
    for key, column in bcif_file["1L2Y"]["audit_author"].items():
        print(f"{key:25}{column.as_array()}")

The main difference is that :class:`BinaryCIFData` has an additional
``encoding`` attribute that specifies how the data is compressed in the binary
representation.
A well chosen encoding can reduce the file size significantly.

.. jupyter-execute::

    import numpy as np

    # Default uncompressed encoding
    array = np.arange(100)
    print(pdbx.BinaryCIFData(array).serialize())
    print("\nvs.\n")
    # Delta encoding followed by run-length encoding
    # [0, 1, 2, ...] -> [0, 1, 1, ...] -> [0, 1, 1, 99]
    print(
        pdbx.BinaryCIFData(
            array,
            encoding = [
                # [0, 1, 2, ...] -> [0, 1, 1, ...]
                pdbx.DeltaEncoding(),
                # [0, 1, 1, ...] -> [0, 1, 1, 99]
                pdbx.RunLengthEncoding(),
                # [0, 1, 1, 99] -> b"\x00\x00..."
                pdbx.ByteArrayEncoding()
            ]
        ).serialize()
    )

As finding good encodings manually can be tedious, :func:`compress()` does this
automatically - from a single :class:`BinaryCIFData` to an entire
:class:`BinaryCIFFile`.

.. jupyter-execute::

    uncompressed_data = pdbx.BinaryCIFData(np.arange(100))
    print(f"Uncompressed size: {len(uncompressed_data.serialize()["data"])} bytes")
    compressed_data = pdbx.compress(uncompressed_data)
    print(f"Compressed size: {len(compressed_data.serialize()["data"])} bytes")


Using structures from a PDBx file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While this low-level API is useful for using the entire potential of
the PDBx format, most applications require only reading/writing a
structure.
As the *BinaryCIF* format is both, smaller and faster to parse, it is
recommended to use it instead of the *CIF* format in *Biotite*.

.. jupyter-execute::

    tc5b = pdbx.get_structure(bcif_file)
    # Do some fancy stuff
    pdbx.set_structure(bcif_file, tc5b)

Similar to :class:`PDBFile`, :func:`get_structure()` creates automatically an
:class:`AtomArrayStack`, even if the file actually contains only a single
model.
If you would like to have an :class:`AtomArray` instead, you have to specify
the :obj:`model` parameter.
