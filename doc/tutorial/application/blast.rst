.. include:: /tutorial/preamble.rst

Finding homologous sequences with BLAST
=======================================

.. currentmodule:: biotite.application.blast

the :mod:`biotite.application.blast` subpackage provides an interface to
*NCBI BLAST*: the :class:`BlastWebApp` class.
Let's dive directly into the code, we try to find homologous sequences to the
miniprotein *TC5b*.

.. Do not run the following Jupyter cells, as BLAST runs may take a long time

.. jupyter-input::

    import biotite.application.blast as blast
    import biotite.sequence as seq

    tc5b_seq = seq.ProteinSequence("NLYIQWLKDGGPSSGRPPPS")
    app = blast.BlastWebApp("blastp", tc5b_seq)
    app.start()
    app.join()
    alignments = app.get_alignments()
    best_ali = alignments[0]
    print(best_ali)
    print()
    print("HSP position in query: ", best_ali.query_interval)
    print("HSP position in hit: ", best_ali.hit_interval)
    print("Score: ", best_ali.score)
    print("E-value: ", best_ali.e_value)
    print("Hit UID: ", best_ali.hit_id)
    print("Hit name: ", best_ali.hit_definition)

.. jupyter-output::

    NLYIQWLKDGGPSSGRPPPS
    NLYIQWLKDGGPSSGRPPPS

    HSP position in query:  (1, 20)
    HSP position in hit:  (1, 20)
    Score:  101
    E-value:  0.000161777
    Hit UID:  1L2Y_A
    Hit name:  Chain A, TC5b [synthetic construct]

This was too simple for BLAST:
As best hit it just found the query sequence itself in the PDB.
However, it gives a good impression about how this :class:`Application` works.
Besides some optional parameters, the :class:`BlastWebApp` requires the BLAST
program and the query sequence.
After the app has finished, you get a list of alignments with descending score.
An alignment is an instance of :class:`BlastAlignment`, a subclass of the
:class:`biotite.sequence.align.Alignment` encountered in a
:doc:`previous chapter <sequence/align_optimal>`.
It contains some additional information as shown above.
The hit UID can be used to obtain the complete hit sequence via
:mod:`biotite.database.entrez`.

The next alignment should be a bit more challenging.
We take a random part of the *E. coli* BL21 genome and distort it a
little bit.
Since we still expect a high similarity to the original sequence,
we decrease the E-value threshold.

.. jupyter-input::

    import biotite.application.blast as blast
    import biotite.sequence as seq

    distorted_bl21_excerpt = seq.NucleotideSequence(
        "CGGAAGCGCTCGGTCTCCTGGCCTTATCAGCCACTGCGCGACGATATGCTCGTCCGTTTCGAAGA"
    )
    app = blast.BlastWebApp("blastn", distorted_bl21_excerpt)
    app.set_max_expect_value(0.1)
    app.start()
    app.join()
    alignments = app.get_alignments()
    best_ali = alignments[0]
    print(best_ali)
    print()
    print("HSP position in query: ", best_ali.query_interval)
    print("HSP position in hit: ", best_ali.hit_interval)
    print("Score: ", best_ali.score)
    print("E-value: ", best_ali.e_value)
    print("Hit UID: ", best_ali.hit_id)
    print("Hit name: ", best_ali.hit_definition)

.. jupyter-output::

    CGGAAGCGCTCGGTCTCCTGGCC---TTATCAGCCACTGCGCGACGATATGCTCGTCCGTTTCGAAGA
    CGGAAGCGCT-GGTC-CCTGCCCGCTTTATCAGGGAATGCGCGACGGCAAAATCGTCCGTTTCGAAGA

    HSP position in query:  (1, 65)
    HSP position in hit:  (4656858, 4656923)
    Score:  56
    E-value:  0.0117615
    Hit UID:  CP026845
    Hit name:  Shigella boydii strain NCTC 9733 chromosome, complete genome