.. include:: /tutorial/preamble.rst

Homology search with MMseqs2
============================

.. currentmodule:: biotite.application_v2.mmseqs

*MMseqs2* is a software suite for fast searching and clustering of large
sequence sets.
Its interface is the :class:`MMseqsApp` class from the
:mod:`biotite.application_v2.mmseqs` subpackage.
In contrast to the MSA programs from the previous chapter, *MMseqs2* is not a
single command, but a collection of subcommands that read and write
*databases*.
Accordingly, :class:`MMseqsApp` has one method per supported subcommand and
each of them returns a :class:`Future` that resolves to a :class:`Database`
object.

Creating databases
------------------
We start with some homologous sequences to search in: a few globins from
different species, plus insulin, which is unrelated to the globins.
Furthermore, we need a query sequence, in our case the human hemoglobin
subunit delta.
Both are fetched from *NCBI Entrez* as FASTA files.

.. jupyter-execute::

    from tempfile import NamedTemporaryFile, gettempdir
    import biotite.database.entrez as entrez

    globin_uids = [
        "P69905", "P01942", "P01966",  # hemoglobin subunit alpha
        "P68871", "P02088", "P02070",  # hemoglobin subunit beta
        "P02144", "P02185",            # myoglobin
        "P01308",                      # insulin
    ]
    temp_file = NamedTemporaryFile(suffix=".fasta", delete=False)
    target_path = entrez.fetch_single_file(
        globin_uids, temp_file.name, "protein", "fasta"
    )
    query_path = entrez.fetch("P02042", gettempdir(), "fa", "protein", "fasta")

*MMseqs2* does not work on FASTA files directly, but on its own database
format.
Hence, the first step is to convert the files into a
:class:`SequenceDatabase` via :meth:`MMseqsApp.create_db()`.

.. jupyter-execute::

    import biotite.application_v2.mmseqs as mmseqs

    app = mmseqs.MMseqsApp()
    target_db = app.create_db(target_path).result()
    query_db = app.create_db(query_path).result()
    print(type(target_db).__name__)

A :class:`Database` is a directory containing the files *MMseqs2* wrote.
By default, this directory is temporary and is deleted when the object is
discarded.
If you want to keep a database, for example to search it again in a later
session, use :meth:`Database.copy()` or :meth:`Database.move()` to put it
into a persistent directory.

Searching a database
--------------------
Now we search the target database for sequences homologous to the query with
:meth:`MMseqsApp.search()`.
By default, *MMseqs2* only computes the score of each hit.
The ``a`` option enables *backtracing*, i.e. the alignment path is stored as
well, which we will need later on to get the actual alignments.
Like in the previous chapter, the command line options are passed as keyword
arguments.

.. jupyter-execute::

    alignment_db = app.search(query_db, target_db, a=True).result()
    print(type(alignment_db).__name__)

The hits are again stored in a database, an :class:`AlignmentDatabase`.
To make them human-readable, *MMseqs2* provides the subcommand to convert them
into a *BLAST*-like table, available via
:meth:`MMseqsApp.convert_alignments()`.
If no output path is given, the table is written into a temporary file.

.. jupyter-execute::

    table_file = app.convert_alignments(
        alignment_db,
        format_mode=mmseqs.AlignmentFormatMode.BLAST_TABLE_WITH_HEADERS,
    ).result()
    with open(table_file.name) as file:
        print(file.read())
    table_file.close()

Only the hemoglobin subunits are found: The sequence identity of the query to
the beta chains is high, to the alpha chains still moderate, but the
myoglobins are too dissimilar for a sequence search with default parameters.
Also note that the identifiers are the accessions *MMseqs2* parsed from the
FASTA headers.

Working with Biotite objects
----------------------------
So far we have used the subcommands as they are: with files as input and
output.
However, usually you have :class:`Sequence` objects at hand and want to get
:class:`Alignment` objects back.
The :mod:`biotite.application_v2.mmseqs` subpackage provides utility
functions for this purpose.
To begin with, :func:`create_database_from_sequences()` creates a
:class:`SequenceDatabase` from a dictionary of sequences.
This way we also get to choose the identifiers ourselves.

.. jupyter-execute::

    import biotite.sequence.io.fasta as fasta

    fasta_file = fasta.FastaFile.read(target_path)
    target_sequences = {
        # Use the UniProt entry name from the header as identifier,
        # e.g. 'HBA_HUMAN'
        header.split("|")[2].split()[0]: sequence
        for header, sequence in fasta.get_sequences(fasta_file).items()
    }
    query_sequence = fasta.get_sequence(fasta.FastaFile.read(query_path))

    target_db = mmseqs.create_database_from_sequences(app, target_sequences)
    query_db = mmseqs.create_database_from_sequences(
        app, {"HBD_HUMAN": query_sequence}
    )
    alignment_db = app.search(query_db, target_db, a=True).result()

Instead of parsing the *BLAST*-like table, :func:`get_alignment_table()`
extracts the columns of interest.

.. jupyter-execute::

    table = mmseqs.get_alignment_table(
        app, alignment_db, ["query", "target", "pident", "evalue"]
    )
    for target, identity, e_value in zip(
        table["target"], table["pident"], table["evalue"]
    ):
        print(f"{target:12}{identity:>8}{e_value:>12}")

Finally, :func:`get_alignments()` reconstructs an :class:`Alignment` for each
hit, which is the reason we enabled backtracing above.
The alignments are ordered by increasing E-value, so the first one is the
best hit.

.. jupyter-execute::

    alignments = mmseqs.get_alignments(app, alignment_db)
    print(list(alignments.keys()))
    print()
    best_alignment = alignments["HBD_HUMAN", "HBB_HUMAN"]
    print(best_alignment)

Clustering
----------
Besides searching, the other main purpose of *MMseqs2* is clustering
sequences by their similarity, in order to reduce redundancy in a sequence
set.
This is done by :meth:`MMseqsApp.cluster()`, that returns a
:class:`ClusterDatabase`, and :func:`get_clusters()`, that maps the
representative sequence of each cluster to its members.
Here we cluster the globins at a minimum sequence identity of 50 %.

.. jupyter-execute::

    cluster_db = app.cluster(target_db, min_seq_id=0.5).result()
    clusters = mmseqs.get_clusters(app, cluster_db)
    for representative, members in clusters.items():
        print(f"{representative}: {members}")

As expected, the alpha chains, the beta chains and the myoglobins form
separate clusters, while insulin is on its own.

Increasing the sensitivity with profiles
----------------------------------------
Remember that the myoglobins were not found by the sequence search.
A common solution for detecting such distant homologs is a *profile search*:
The hits from the first search are summarized into a sequence profile via
:meth:`MMseqsApp.result_to_profile()`, that captures the conservation of each
position of the query.
The resulting :class:`ProfileDatabase` can be used as query for another search
in place of the sequence database.
Combined with a high sensitivity, requested via the ``s`` option, the
myoglobins are found this time.

.. jupyter-execute::

    profile_db = app.result_to_profile(alignment_db).result()
    profile_alignment_db = app.search(
        profile_db, target_db, a=True, s=7.5
    ).result()
    table = mmseqs.get_alignment_table(
        app, profile_alignment_db, ["target", "pident", "evalue"]
    )
    for target, identity, e_value in zip(
        table["target"], table["pident"], table["evalue"]
    ):
        print(f"{target:12}{identity:>8}{e_value:>12}")

If you are interested in the profile itself, :func:`get_matrix_from_profile()`
converts it into a position-specific :class:`SubstitutionMatrix`.

Structure search with Foldseek
------------------------------
*Foldseek* applies the same concepts to protein structures:
It converts structures into sequences of structural states and then uses the
machinery of *MMseqs2* to search and cluster them.
Consequently, its interface :class:`FoldseekApp` provides the same methods and
databases, only that :func:`create_database_from_structures()` creates the
database from :class:`AtomArray` objects instead of sequences.
