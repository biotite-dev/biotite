.. include:: /tutorial/preamble.rst

Pairwise sequence alignment
===========================

.. currentmodule:: biotite.sequence.align

When comparing two (or more) sequences, usually an alignment needs to be
performed.
Two kinds of algorithms need to be distinguished here:
Heuristic algorithms do not guarantee to yield the optimal alignment, but
instead they are very fast.
This approach will be used in a :doc:`later chapter <align_heuristic>`.
On the other hand, there are algorithms that calculate the optimal alignment,
i.e. they find the arrangement of symbols with the highest similarity score,
but are quite slow.

The :mod:`biotite.sequence.align` package provides the function
:func:`align_optimal()`, which fits into the latter category.
It either performs an optimal global alignment, i.e. the entire sequences are
aligned, or an optimal local alignment, i.e. only the highest scoring excerpt
of both sequences are aligned.

:func:`align_optimal()` (as well as most other functionalities in
:mod:`biotite.sequence.align`) can align any two
:class:`Sequence` objects with each other.
In fact the :class:`Sequence` objects can be instances from different
:class:`Sequence` subclasses and therefore may have different
alphabets, although in the overwhelming number of use cases the alphabets
of the aligned sequences will be the same.
The only condition that must be satisfied, is that the
:class:`SubstitutionMatrix` alphabets match the alphabets of the
sequences to be aligned:
A :class:`SubstitutionMatrix` maps a combination of two symbols, one from the
first sequence the other one from the second sequence, to a similarity score.
A :class:`SubstitutionMatrix` object contains two alphabets with
length *n* or *m*, respectively, and an *(n,m)*-shaped
:class:`ndarray` storing the similarity scores.
You can choose one of many predefined matrices from an internal
database or you can create a custom matrix on your own.

So much for theory.
Let's start by showing different ways to construct a
:class:`SubstitutionMatrix`, in our case for protein sequence
alignments:

.. jupyter-execute::

    import biotite.sequence as seq
    import biotite.sequence.align as align
    import numpy as np

    alph = seq.ProteinSequence.alphabet
    # Load the standard protein substitution matrix, which is BLOSUM62
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    print("\nBLOSUM62\n")
    print(matrix)
    # Load another matrix from internal database
    matrix = align.SubstitutionMatrix(alph, alph, "BLOSUM50")
    # Load a matrix dictionary representation,
    # modify it, and create the SubstitutionMatrix
    # (The dictionary could be alternatively loaded from a string containing
    # the matrix in NCBI format)
    matrix_dict = align.SubstitutionMatrix.dict_from_db("BLOSUM62")
    matrix_dict[("P","Y")] = 100
    matrix = align.SubstitutionMatrix(alph, alph, matrix_dict)
    # And now create a matrix by directly providing the ndarray
    # containing the similarity scores
    # (identity matrix in our case)
    scores = np.identity(len(alph), dtype=int)
    matrix = align.SubstitutionMatrix(alph, alph, scores)
    print("\n\nIdentity matrix\n")
    print(matrix)

For our protein sequence alignment we will use the standard *BLOSUM62* matrix.
The final decision we need to make is the gap penalty:
It defines the score penalty for inserting a gap into the alignment.
Here we can choose between a linear gap penalty (same penalty for each gap) and
an affine gap penalty (high penalty for gap opening, low penalty for extension
of a gap).

.. jupyter-execute::

    seq1 = seq.ProteinSequence("BIQTITE")
    seq2 = seq.ProteinSequence("IQLITE")
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    print("\nLocal alignment")
    alignments = align.align_optimal(
        # A single integer indicates a linear gap penalty
        seq1, seq2, matrix, gap_penalty=-10, local=True
    )
    for ali in alignments:
        print(ali)
    print()
    print("Global alignment")
    alignments = align.align_optimal(
        seq1, seq2, matrix, gap_penalty=-10, local=False
    )
    for ali in alignments:
        print(ali)

As you might have noticed, the function does not return a single alignment,
but an entire list.
The reason is that there can be multiple optimal alignments with the same
score.
Each alignment is represented by an :class:`Alignment` object.
This object saves the input sequences together with a so called trace
- the indices to symbols in these sequences that are aligned to each other
(``-1`` for a gap).
Additionally the alignment score is stored in this object.
This object can also prettyprint the alignment into a human readable form as
demonstrated above.

For publication purposes you can create an actual figure using *Matplotlib*.
You can either decide to color the symbols based on the symbol type
or based on the similarity within the alignment columns.
In this case we will go with the similarity visualization.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import biotite.sequence.graphics as graphics

    fig, ax = plt.subplots(figsize=(2.0, 0.8), constrained_layout=True)
    graphics.plot_alignment_similarity_based(
        ax, alignments[0], matrix=matrix, symbols_per_line=len(alignments[0])
    )

If you are interested in more advanced visualization examples, have a
look at the
:doc:`example gallery <../../examples/gallery/sequence/homology/index>`.

You can also do some simple analysis on these objects, like
determining the sequence identity or calculating the score.
For further custom analysis, it can be convenient to have directly the
aligned symbols codes instead of the trace.

.. jupyter-execute::

    alignment = alignments[0]
    print("Score: ", alignment.score)
    print("Recalculated score:", align.score(alignment, matrix=matrix))
    print("Sequence identity:", align.get_sequence_identity(alignment))
    print("Symbols:")
    print(align.get_symbols(alignment))
    print("Symbol codes:")
    print(align.get_codes(alignment))

Loading alignments from FASTA files
-----------------------------------

.. currentmodule:: biotite.sequence.io.fasta

You might wonder, why you should recalculate the score, when the score
has already been directly computed via :func:`align_optimal()`.
The answer is that you might load an alignment from a FASTA file
using :func:`get_alignment()`, where the score is not provided.

.. jupyter-execute::

    from tempfile import NamedTemporaryFile
    import biotite.sequence.io.fasta as fasta

    temp_file = NamedTemporaryFile(suffix=".fasta", delete=False)
    fasta_file = fasta.FastaFile()
    fasta.set_alignment(fasta_file, alignment, seq_names=["seq_1", "seq_2"])
    print(fasta_file)
    fasta_file.write(temp_file.name)

    fasta_file = fasta.FastaFile.read(temp_file.name)
    alignment = fasta.get_alignment(fasta_file)
    alignment.score = align.score(alignment, matrix=matrix)
