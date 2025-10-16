.. include:: /tutorial/preamble.rst

Accessing sequence data in NCBI Entrez
======================================

.. currentmodule:: biotite.database.entrez

An important source of biological sequences including their annotations is the
*NCBI Entrez* database, which is commonly known as '*the NCBI*'.
To download data we need to provide the *unique record identifier*
(UID) of the entry.
This can either be the *Accession* or *GI*, which are parallel identification
systems.
Furthermore, we need

- the database name from which we would like to download the record, which can either
  be the internal name (e.g. ``'nuccore'``) or the user-facing name
  (e.g. ``'Nucleotide'``),
- and the retrieval type, which is the file format of the downloaded data
  (e.g. ``'fasta'``).

A list of valid combinations can be found
`here <https://www.ncbi.nlm.nih.gov/books/NBK25499/table/chapter4.T._valid_values_of__retmode_and/?report=objectonly>`_.
In the following case we will download the protein sequence of hemoglobin.

.. jupyter-execute::

    from tempfile import gettempdir, NamedTemporaryFile
    import biotite.database.entrez as entrez

    file_path = entrez.fetch(
        "6BB5_A", gettempdir(), suffix="fa",
        db_name="protein", ret_type="fasta"
    )
    with open(file_path) as file:
        print(file.read())

Note the ``subunit alpha`` in the header of the FASTA file:
Hemoglobin is a tetramer, consisting of two alpha and two beta subunits.
Hence, we will download the sequence of the beta subunit as well.
We can download multiple records at once by providing a list of UIDs.
In addition, now we are also interested in sequence annotation, like sequence
ranges where some secondary structure is present.
Therefore, we want to download the data in *GenBank* format.

.. jupyter-execute::

    from os.path import basename

    file_paths = entrez.fetch(
        ["6BB5_A", "6BB5_B"], gettempdir(), suffix="fa",
        db_name="protein", ret_type="gb"
    )
    print([basename(path) for path in file_paths])

File formats like *GenBank* or *FASTA* allow multiple records in a single file.
Downloading such multi-record files is also possible.

.. jupyter-execute::

    temp_file = NamedTemporaryFile(suffix=".fasta", delete=False)
    file_path = entrez.fetch_single_file(
        ["6BB5_A", "6BB5_B"], temp_file.name,
        db_name="protein", ret_type="fasta"
    )
    with open(file_path) as file:
        print(file.read())

Searching for records
---------------------
Only rarely we know the UID of the record we are looking for upfront.
Usually one has only some criteria, such as the name of a gene or the organism.
:mod:`biotite.database.entrez` allows searching for UIDs satisfying certain
criteria.
The obtained list of UIDs can then be used to download the records as shown
above.

.. jupyter-execute::

    # Search the Nucleotide database in all fields for the term "Lysozyme"
    print(entrez.search(entrez.SimpleQuery("Lysozyme"), db_name="nuccore"))

:func:`search()` takes a :class:`Query` and returns a list of UIDs.
Note that by default only 20 results are returned.
To increase or decrease this value, you can adjust the ``number`` parameter.

Instead of searching in all fields, we can also search for a term in a specific
field.
Furthermore, we can logically combine multiple :class:`Query` objects using
``|``, ``&`` and ``^``, that represent ``OR``,  ``AND`` and ``NOT`` linkage,
respectively.
The :class:`Query` can be converted into a string representation, that would
also work in search bar on the NCBI website.

.. jupyter-execute::

    composite_query = (
        entrez.SimpleQuery("50:100", field="Sequence Length") &
        (
            entrez.SimpleQuery("Escherichia coli", field="Organism") |
            entrez.SimpleQuery("Bacillus subtilis", field="Organism")
        )
    )
    print(composite_query)
    print(entrez.search(composite_query, db_name="nuccore"))

Increasing the request limit
----------------------------
The NCBI Entrez database has a quite conservative request limit.
Hence, frequent accesses to the database may raise a ``RequestError``.
The limit can be greatly increased by providing an
`NCBI API key <https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/>`_.

.. jupyter-execute::

    api_key = "api_key_placeholder"
    entrez.set_api_key(api_key)