Big data - The Database subpackage
----------------------------------

Biological databases are the backbone of computational biology. The
``database`` subpackage provides interfaces for popular online databases
like the RCSB PDB or the NCBI Entrez database.

Fetching structure files from the RCSB PDB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Downloading structure files from the RCSB PDB is quite easy: Simply specify
the PDB ID, the file format and the target directory for the `fetch()` function
and you are done. The function even returns the path to the downloaded file,
so you can just load it via the ``structure.io`` package.

.. code-block:: python

   import biopython.database.rcsb as rcsb
   import biopython.structure.io as strucio
   file_path = rcsb.fetch("1l2y", "cif", "path/to/directory")
   atom_array = strucio.get_structure_from(file_path)

Often you just need a file temporarily for loading it into your script. In this
case the `Biopython` temporary directory is recommended. The temporary
directory will be present as long as the script is running. Afterwards the
directory is automatically deleted.

.. code-block:: python

   import biopython
   file_path = rcsb.fetch("1l2y", "cif", biopython.temp_dir())

In case you want to dowload multiple files, you are able to specify a list
of PDB IDs, which in return gives you a list of file_paths.

.. code-block:: python

   file_paths = rcsb.fetch(["1l2y", "3o5r], "cif", biopython.temp_dir())

By default `fetch()` checks wether the file to be fetched already exists
in the directory, and does download it if it does not exist yet. If you want to
download files irrespectively, set `overwrite` to true.

.. code-block:: python

   file_path = rcsb.fetch("1l2y", "cif", "path/to/directory", overwrite=True)

Another feature is the search ability, which interfaces RCSB's SEARCH service.
At first you have to create `Query` object for the property you want to filer.
The `search()` method takes the `Query` and returns a list of PDB IDs.


Different queries can be 'and'/'or' combined using a `CompositeQuery`.



Not all query types of the SEARCH service are supported yet. But it is quite
easy to implement your needed query type by inheriting `SimpleQuery`. The
API reference contains more information on that.

Fetching files from the NCBI Entrez database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another important source of biological information is the NCBI Entrez database,
which is commonly known as 'the NCBI'. It provides a myriad of information,
ranging from sequences and sequence features to scientific papers. Fetching
files from NCBI Entrez works quite similar to the RCSB interface:

