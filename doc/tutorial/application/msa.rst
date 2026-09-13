.. include:: /tutorial/preamble.rst

Multiple sequence alignments
============================

.. currentmodule:: biotite.application_v2.muscle

The :mod:`biotite.application_v2` subpackage provides interfaces to various
*multiple sequence alignments* (MSAs) programs.
For our example we choose the software MUSCLE:
The subpackage :mod:`biotite.application_v2.muscle` contains the class
:class:`Muscle3App` that does the job.
First we get some homologous input sequences from *NCBI Entrez*.

.. jupyter-execute::

    from tempfile import NamedTemporaryFile
    import biotite.sequence.io.fasta as fasta
    import biotite.database.entrez as entrez

    # Use cyclotide sequences again, but this time more than two
    query = (
        entrez.SimpleQuery("Cyclotide") &
        entrez.SimpleQuery("cter") &
        entrez.SimpleQuery("srcdb_swiss-prot", field="Properties") ^
        entrez.SimpleQuery("Precursor")
    )
    uids = entrez.search(query, "protein")
    temp_file = NamedTemporaryFile(suffix=".fasta", delete=False)
    fasta_file = fasta.FastaFile.read(
        entrez.fetch_single_file(uids, temp_file.name, "protein", "fasta")
    )
    sequences = list(fasta.get_sequences(fasta_file).values())

We simply input the sequences to :meth:`Muscle3App.run()`, wait for the
application to finish and get the resulting :class:`Alignment` object from
the :class:`Muscle3Result`.

.. jupyter-execute::

    import biotite.application_v2.muscle as muscle

    app = muscle.Muscle3App()
    result = app.run(sequences).result()
    alignment = result.alignment
    print(alignment)

In most MSA software even more information than the mere alignment can be
extracted.
For instance the guide tree, that was used for the alignment, is also part
of the result.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import biotite.sequence.graphics as graphics

    tree = result.guide_tree_identity

    fig, ax = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    graphics.plot_dendrogram(ax, tree, orientation="left", show_distance=True)
    _ = ax.set_xlabel("Distance")

If you have MUSCLE version 5 installed, use :class:`Muscle5App` instead,
which is used in the same way.

Variety of MSA software
-----------------------
The alternatives to MUSCLE are Clustal Omega and MAFFT.
To use them, simply replace :class:`Muscle3App` with
:class:`~biotite.application_v2.clustalo.ClustalOmegaApp` or
:class:`~biotite.application_v2.mafft.MafftApp`.

.. jupyter-execute::

    import biotite.application_v2.clustalo as clustalo

    alignment = clustalo.ClustalOmegaApp().run(sequences).result().alignment
    print(alignment)

As shown in the output, the alignment with Clustal Omega slightly
differs from the one performed with MUSCLE.

Custom substitution matrices and sequence types
-----------------------------------------------
If the MSA software supports protein sequence alignment **and** custom
substitution matrices, e.g. MUSCLE and MAFFT, almost any type
of sequence can be aligned.
Let's show this on the example of a nonsense alphabet.

.. jupyter-execute::

    import numpy as np
    import biotite.application_v2.mafft as mafft
    import biotite.sequence as seq
    import biotite.sequence.align as align

    alphabet = seq.Alphabet(("foo", "bar", 42))
    sequences = [
        seq.GeneralSequence(alphabet, symbols) for symbols in [
            ["foo", "bar", 42, "foo", "foo", 42, 42],
            ["foo", 42, "foo", "bar", "foo", 42, 42],
        ]
    ]
    matrix = align.SubstitutionMatrix(
        alphabet, alphabet, np.array([
            [ 100, -100, -100],
            [-100,  100, -100],
            [-100, -100,  100]
        ])
    )
    result = mafft.MafftApp().run(sequences, matrix=matrix).result()
    alignment = result.alignment
    # As the alphabet does not have characters as symbols
    # the alignment cannot be directly printed
    # However, we can print the trace
    print(alignment.trace)

This works, because internally the sequences and the matrix are converted into
protein sequences/matrix.
Then the masquerading sequences are aligned via the software and
finally the sequences are mapped back into the original sequence type.
