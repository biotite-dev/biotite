.. image:: https://raw.githubusercontent.com/biotite-dev/fastpdb/main/logo.svg
    :width: 300
    :align: center
    :alt: fastpdb


|
|

A high performance drop-in replacement for *Biotite*'s ``PDBFile``
written in *Rust*.

Installation
------------

``fastpdb`` can be installed via

.. code-block:: console

    $ pip install fastpdb

Usage
-----

You can simply replace ``biotite.structure.io.pdb.PDBFile`` by
``fastpdb.PDBFile``. The methods and their parameters are the same.

.. code-block:: python

    import fastpdb

    in_file = fastpdb.PDBFile.read("path/to/file.pdb")
    atom_array = in_file.get_structure(model=1)

    out_file = fastpdb.PDBFile()
    out_file.set_structure(atom_array)
    out_file.write("path/to/another_file.pdb")

Note that ``fastpdb`` does not yet support the *hybrid-36* PDB format.


Performance
-----------

``fastpdb`` is multiple times faster than ``biotite``.

.. image:: https://raw.githubusercontent.com/biotite-dev/fastpdb/main/benchmark.svg
    :width: 800
    :align: center
