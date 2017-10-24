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
The `search()` method takes the `Query` and returns a list of PDB IDs, which
itself can be used as inpt for `fetch()`.

.. code-block:: python

   query = rcsb.ResolutionQuery(0.0, 0.6)
   pdb_ids = rcsb.search(query)
   print(pdb_ids)
   files = rcsb.fetch(pdb_ids, "cif", temp_dir())
   for file in files:
       array = strucio.get_structure_from(file)
       # Do some fancy stuff

Output:

.. code-block:: none

   ['1EJG', '1I0T', '3NIR', '3P4J', '5D8V', '5NW3']

Multiple queries can be 'and'/'or' combined using a `CompositeQuery`.

.. code-block:: python

   query1 = rcsb.ResolutionQuery(0.0, 1.0)
   query2 = rcsb.MolecularWeightQuery(10000, 100000)
   composite = rcsb.CompositeQuery("and", [query1, query2])

Not all query types of the SEARCH service are supported yet. But it is quite
easy to implement your needed query type by inheriting `SimpleQuery`. The
API reference contains more information on that.

Fetching files from the NCBI Entrez database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another important source of biological information is the NCBI Entrez database,
which is commonly known as 'the NCBI'. It provides a myriad of information,
ranging from sequences and sequence features to scientific papers. Fetching
files from NCBI Entrez works analogous to the RCSB interface. This time
we have to provide the database specific UIDs instead of PDB IDs:

.. code-block:: python
   
   import biopython.database.entrez as entrez
   import biopython.sequence.io.fasta as fasta
   import biopython
   files = entrez.fetch(["1L2Y_A","3O5R_A"], biopython.temp_dir(), suffix="fa",
                 db_name="protein", ret_type="fasta")
   for file in files:
       fasta_file = fasta.FastaFile()
       fasta_file.read(file)
       prot_seq = fasta.get_sequence(fasta_file)
       # Do also some fancy stuff here

A list of valid database, retrieval type and mode combinations can
be found
`here <https://www.ncbi.nlm.nih.gov/books/NBK25499/table/chapter4.T._valid_values_of__retmode_and/?report=objectonly>`.
The Entrez database allows for packing data for multiple UIDs into a single
file. This is achieved with the `fetch_single_file()` function.

.. code-block:: python
   
   file = entrez.fetch_single_file(["1L2Y_A","3O5R_A"],
                                   biopython.temp_file("sequences.fa"),
                                   db_name="protein", ret_type="fasta")
