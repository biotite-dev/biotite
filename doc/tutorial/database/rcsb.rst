.. include:: /tutorial/preamble.rst

Fetching structure files from the RCSB PDB
==========================================

.. currentmodule:: biotite.database.rcsb

Downloading structure files from the *RCSB PDB* is quite easy:
We simply specify the PDB ID, the file format and the target directory
for the :func:`fetch()` function.
The function returns the path to the downloaded file, so you
can simply load the file via the :mod:`biotite.structure.io` subpackage
(more on this in a :doc:`later tutorial <../structure/formats>`).
We will download on a protein structure of the miniprotein *TC5b*
(PDB: 1L2Y) into a temporary directory.

.. jupyter-execute::

    from tempfile import gettempdir
    from os.path import basename
    import biotite.database.rcsb as rcsb

    file_path = rcsb.fetch("1l2y", "pdb", gettempdir())
    print(basename(file_path))

In case we want to download multiple files, we are able to specify a
list of PDB IDs, which in return gives us a list of file paths.

.. jupyter-execute::

    # Download files in the more modern mmCIF format
    file_paths = rcsb.fetch(["1l2y", "1aki"], "cif", gettempdir())
    print([basename(file_path) for file_path in file_paths])

By default :func:`fetch()` checks whether the file to be fetched
already exists in the directory and downloads it, if it does not
exist yet.
If we want to download files irrespectively, set :obj:`overwrite` to
true.

.. jupyter-execute::

    # Download file in the fast and small BinaryCIF format
    file_path = rcsb.fetch("1l2y", "bcif", gettempdir(), overwrite=True)

If we omit the file path or set it to ``None``, the downloaded data
will be returned directly as a file-like object, without creating a
file on the disk at all.

.. jupyter-execute::

    file = rcsb.fetch("1l2y", "pdb")
    lines = file.readlines()
    print("".join(lines[:10] + ["..."]))

Searching for entries
---------------------
As mentioned in the previous chapter, in many cases one is not interested in a
specific structure, but in set of structures that fits some desired criteria.
And also similar to the other :mod:`biotite.database` subpackages,
PDB IDs matching those criteria can be searched for by defining a
:class:`Query` and passing it to :func:`search()`.
For this purpose the *RCSB* search API can be used.
Likewise, :func:`count()` is used to request the number of matching
PDB IDs, which is faster and more database-friendly than measuring the length
of the list returned by a :func:`search()` call.

.. jupyter-execute::

    query = rcsb.BasicQuery("HCN1")
    pdb_ids = rcsb.search(query)
    print(pdb_ids)
    print(rcsb.count(query))
    files = rcsb.fetch(pdb_ids, "cif", gettempdir())

This was a simple search for the occurrence of the search term in any
field.
You can also search for a value in a specific field with a
:class:`FieldQuery`.
A complete list of the available fields and its supported operators
is documented
`on this page <https://search.rcsb.org/structure-search-attributes.html>`_
and `on that page <https://search.rcsb.org/chemical-search-attributes.html>_`.

.. jupyter-execute::

    # Query for 'lacA' gene
    query1 = rcsb.FieldQuery(
        "rcsb_entity_source_organism.rcsb_gene_name.value",
        exact_match="lacA"
    )
    # Query for resolution below 1.5 Å
    query2 = rcsb.FieldQuery("reflns.d_resolution_high", less=1.5)

The search API allows even more complex queries, e.g. for sequence
or structure similarity. Have a look at the API reference of
:mod:`biotite.database.rcsb`.

Multiple :class:`Query` objects can be combined using the ``|`` (or)
or ``&`` (and) operator for a more fine-grained selection.
A :class:`FieldQuery` is negated with ``~``.

.. jupyter-execute::

    composite_query = query1 & ~query2
    print(rcsb.search(composite_query))

Often the structures behind the obtained PDB IDs have degree of
redundancy.
For example they may represent the same protein sequences or result
from the same set of experiments.
You may use :class:`Grouping` of structures to group redundant
entries or even return only single representatives of each group.

.. jupyter-execute::

    query = rcsb.BasicQuery("Transketolase")
    # Group PDB IDs from the same collection
    print(rcsb.search(
        query, group_by=rcsb.DepositGrouping(), return_groups=True
    ))
    # Get only a single representative of each group
    print(rcsb.search(
        query, group_by=rcsb.DepositGrouping(), return_groups=False
    ))

Note that grouping may omit PDB IDs in search results, if such PDB IDs
cannot be grouped.
For example in the case shown above only a few PDB entries were
uploaded as collection and hence are part of the search results.

Getting computational models
----------------------------
By default :func:`search()` only returns experimental structures.
In addition to that the RCSB lists an order of magnitude more computational models.
They can be included in search results by adding ``"computational"`` to the
``content_types`` parameter.

.. jupyter-execute::

    query = (
        rcsb.FieldQuery("rcsb_polymer_entity.pdbx_description", contains_phrase="Lysozyme")
        & rcsb.FieldQuery(
            "rcsb_entity_source_organism.scientific_name", exact_match="Homo sapiens"
        )
    )
    ids = rcsb.search(query, content_types=("computational",))
    print(ids)

The returned four-character IDs are the RCSB PDB IDs of experimental structures
like we already saw above.
The IDs with the ``AF_`` on the other hand are computational models from
*AlphaFold DB*.

.. currentmodule:: biotite.database.afdb

To download those we require another subpackage: :mod:`biotite.database.afdb`.
Its :func:`fetch()` function works very similar.

.. jupyter-execute::

    import biotite.database.afdb as afdb

    files = []
    # For the sake of run time, only download the first 5 entries
    for id in ids[:5]:
        if id.startswith("AF_"):
            # Entry is in AlphaFold DB
            files.append(afdb.fetch(id, "cif", gettempdir()))
        elif id.startswith("MA_"):
            # Entry is in ModelArchive, which is not yet supported
            raise NotImplementedError
        else:
            # Entry is in RCSB PDB
            files.append(rcsb.fetch(id, "cif", gettempdir()))
    print([basename(file) for file in files])