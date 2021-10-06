.. raw:: html

    <p align="center">
       <img width="300px" src="logo.svg" alt="fastpdb">
    </p>


|
|

A high performance drop-in replacement for *Biotite*'s ``PDBFile``
written in *Rust*.

Installation
------------

``fastpdb`` can be installed via

.. code-block:: console

    $ pip install fastpdb

or

.. code-block:: console

    $ conda install -c conda-forge fastpdb


Usage
-----

You can simply replace ``biotite.structure.io.pdb.PDBFile`` by
``fastpdb.PDBFile``. The methods and their parameters are the same.

.. code-block:: python

    import fastpdb

    in_file = fastpdb.PDBFile.read("1AKI.pdb")
    atom_array = in_file.get_structure(model=1)

    out_file = fastpdb.PDBFile()
    out_file.set_structure(atom_array)
    out_file.write("test.pdb")


Performance
-----------

``fastpdb`` is multiple times faster compared to ``biotite``.

.. raw:: html

    <p align="center">
       <img width="800px" src="benchmark.svg">
    </p>