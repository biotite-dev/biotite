.. include:: /tutorial/preamble.rst

Heuristic sequence alignments
=============================

.. currentmodule:: biotite.sequence.align

While the alignment method presented in the previous chapter returns the
optimal alignment of two sequences, it is not recommended to use this method
in scenarios where a either many sequences (e.g. an entire database) or
long sequences (e.g. a genome) is involved:
The computation time and memory space requirements scale
linearly with the length of both sequences, so even if your RAM does
not overflow, you might need to wait a very long time for your
alignment results.

To remedy this problem a lively zoo of *heuristic* alignment methods
(aka. *alignment searches*) have surfaced in the last decades.
These methods may not yield the optimal alignment or even miss a similar region
between two sequences entirely,
but in most cases they are sufficiently accurate and they run often multiple
orders of magnitude faster.
Usually heuristic approaches use multiple stages:
The initial stages run fast at low specificity, i.e. they find a lot of matches
between two sequences, but most of them are spurious.
The later stages apply increasingly accurate (and computationally expensive)
filters on these initial matches until only significant sequence alignments
remain.

*Biotite* provides a modular system to build such an alignment search
method yourself, by letting you combine separate functionalities into the
aforementioned multi-stage process.
In the following, we build our own simple alignment search method and apply it
on two toy sequences:
one representing a reference sequence, that could be substituted by a sequence
database or a genome in a real-world scenario, and the other one representing a
query sequence, that a user would give as input to the alignment program.
The latter could be substituted by a sequence database or a genome in a
real-world scenario.

Note that although the setup of such a heuristic alignment is more involved
than the optimal alignment approach presented before, the performance of the
heuristic method scales much better for large sequence data.

.. jupyter-execute::

    import numpy as np
    import biotite.sequence as seq
    import biotite.sequence.align as align

    # Cyclotide F
    reference = seq.ProteinSequence("GIPCGESCVFIPCISSVVGCSCKSKVCYLD")
    # Cyclotide E
    query = seq.ProteinSequence("GIPCAESCVWIPCTVTALLGCSCKDKVCYLD")

    # This is the alignment we would expect in the end
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    print(align.align_optimal(reference, query, matrix)[0])


Stage 1: k-mer matching
-----------------------
A popular approach to alignment search is *k-mer matching*, which is used
by prominent software such as *BLAST* or *MMseqs*.
The *k-mers* of a sequence are all subsequences with length *k*.
The :class:`KmerAlphabet` class provides a way to enumerate all *k-mers*
in a sequence.

.. jupyter-execute::

    import biotite.sequence.align as align

    # We want to list all 3-mers
    kmer_alphabet = align.KmerAlphabet(base_alphabet=query.alphabet, k=3)
    kmers = kmer_alphabet.decode_multiple(kmer_alphabet.create_kmers(query.code))
    print(query, "\n")
    for i, kmer in enumerate(kmers):
        # Print in a way that the k-mers are aligned
        print(" " * i + "".join(kmer))

To find matches between two sequences efficiently, *Biotite* provides the
:class:`KmerTable` class:
It maps each possible *k-mer* to their positions in the given input
reference sequence(s).

.. jupyter-execute::

    # Create a k-mer index table from the k-mers of the reference sequence
    kmer_table = align.KmerTable.from_sequences(
        # Use 3-mers
        k=3,
        # Add only the reference sequence to the table
        sequences=[reference],
        # The purpose of the reference ID is to identify the sequence
        # As there is only one sequence in this simple example,
        # there is only one ID
        ref_ids=[0]
    )
    for ref_id, position in kmer_table[kmer_alphabet.encode("IPC")]:
        print(position)

To get matching positions of a query in the reference, the :class:`KmerTable`
only needs to enumerate all *k-mers* in the query and look up the positions
in the table.
The decision for *k* is important here:
A small *k* improves the sensitivity, but gives also a high number of spurious
matches that must be filtered out later at the cost of computation time.
In contrast a large *k* will directly only find longer identical stretches,
but thus might miss shorter homologous regions.
In literature you can find recommendations for *k*.
In this simple case we choose rather short 3-mers.

.. jupyter-execute::

    matches = kmer_table.match(query)

    # Print matching sequence positions as a matrix
    def print_match_matrix(reference, query, matches):
        match_matrix = np.zeros((len(reference), len(query)), dtype=bool)
        for query_pos, ref_id, ref_pos in matches:
            match_matrix[ref_pos, query_pos] = True
        print("  " + str(reference))
        for i in range(len(query)):
            print(query[i] + " " + " ".join(
                "#" if match_matrix[j, i] else "" for j in range(len(reference))
            ))

    print_match_matrix(reference, query, matches)

These are quite a bunch of matches.
However, they are mostly appearing in consecutive positions in both sequences,
for the simple reason that the stretches, where both sequences match, are longer
than *k*.
We say that these matches are on the same *diagonal*.
Often one is interested only in one alignment per diagonal, as the later stages
will create the same alignments for matches on the same diagonal.
Therefore, we keep only one match per diagonal.

.. jupyter-execute::

    # The diagonal is defined by the difference between the positions in the
    # reference and query sequence.
    diagonals = matches[:,2] - matches[:,0]
    unique_diagonals, unique_indices = np.unique(diagonals, return_index=True)
    matches = matches[unique_indices]
    print_match_matrix(reference, query, matches)


Stage 2: Ungapped seed extension
--------------------------------
The typical next step is an ungapped seed extension.
This means an alignment is created without introducing gaps expanding from a
match position, which can be computed much faster than a gapped alignment.
The ungapped alignment stops when the alignment score drops below a certain
threshold below the maximum score already encountered.
In other words the alignment comprises only a region that is decently similar.

.. jupyter-execute::

    THRESHOLD = 20

    matrix = align.SubstitutionMatrix.std_protein_matrix()
    alignments = []
    for query_pos, ref_id, ref_pos in matches:
        alignment = align.align_local_ungapped(
            reference, query, matrix,
            seed=(ref_pos, query_pos), threshold=THRESHOLD
        )
        alignments.append(alignment)

    for alignment in alignments:
        print("Score:", alignment.score)
        print(alignment)
        print("\n")

Now we can reduce the number of alignments for the next stage, by requiring a
minimum similarity score, to keep only the most promising ones for the upcoming
costly gapped sequence alignment.
For the purpose of this tutorial this threshold is arbitrarily chosen.

.. jupyter-execute::

    SCORE_THRESHOLD = 30

    matches = matches[[ali.score >= SCORE_THRESHOLD for ali in alignments]]

Stage 3: Gapped sequence alignment
----------------------------------
The difference between the heuristic gapped sequence alignment methods and
:meth:`align_optimal()` is that the former only ideally traverses through a
small fraction of the possible alignment search space, allowing them to run
much faster.
However, like :meth:`align_local_ungapped()` they need to be informed with a
match position to start from.
Furthermore, in some cases they might not find the optimal alignment, when the
assumption of the method does not hold for such alignment.
In this tutorial we will use :func:`align_banded()`:
It subsets the alignment space to a (narrow) diagonal band around the match
position.
This means, that the method assumes that the alignment does not contain a
large number of gaps.
If the optimal alignment would contain many gaps, such alignment would not be
found.

.. jupyter-execute::

    BAND_WIDTH = 4

    matrix = align.SubstitutionMatrix.std_protein_matrix()
    alignments = []
    for query_pos, ref_id, ref_pos in matches:
        diagonal = query_pos - ref_pos
        alignment = align.align_banded(
            reference, query, matrix, gap_penalty=-5, max_number=1,
            # Center the band at the match diagonal and extend the band by
            # one half of the band width in each direction
            band=(diagonal - BAND_WIDTH//2, diagonal + BAND_WIDTH//2)
        )[0]
        alignments.append(alignment)

    for alignment in alignments:
        print(alignment)
        print("\n")

The first two alignments are actually duplicates.
The reason is they were initiated from the two distinct diagonals, as the
matches before and after the gap are on different diagonals.

Stage 4: Significance evaluation
--------------------------------
Now we have obtained multiple alignments, but which one of them is the
'correct' one?
in this simple example, we could simply select the one with the highest
similarity score, but this approach is not sound in general:
A reference sequence might contain multiple regions, that are homologous to the
query, or none at all.
A better approach is a statistical measure, like the
`BLAST E-value <https://www.ncbi.nlm.nih.gov/BLAST/tutorial/Altschul-1.html>`_.
It gives the number of alignments expected by chance with a score at least as
high as the score obtained from the alignment of interest.
Hence, a value close to zero means a very significant homology.
Above 0.05 one typically rejects the alignment as insignificant.

We can calculate the E-value using the :class:`EValueEstimator`, that needs to
be initialized with the same scoring scheme used for our alignments.
For the sake of simplicity we choose uniform background frequencies for each
symbol, but usually you would choose values that reflect the amino
acid/nucleotide composition in your sequence database.

.. jupyter-execute::

    estimator = align.EValueEstimator.from_samples(
        seq.ProteinSequence.alphabet, matrix, gap_penalty=-5,
        frequencies=np.ones(len(seq.ProteinSequence.alphabet)),
        # Trade accuracy for a shorter run time for this tutorial
        sample_length=200
    )

Now we can calculate the E-value for the alignments.
Since we have aligned the query only to the reference sequence shown
above, we pass its length to :meth:`EValueEstimator.log_evalue`.
If you have an entire sequence database you align against, you would
take the total sequence length of the database instead.

.. jupyter-execute::

    scores = [alignment.score for alignment in alignments]
    evalues = 10 ** estimator.log_evalue(scores, len(query), len(reference))
    for alignment, evalue in zip(alignments, evalues):
        print(f"E-value = {evalue:.2e}")
        print(alignment)
        print("\n")

The results show that only one alignment is significant:
the same one we have found with :meth:`align_optimal()` above.

Conclusion
----------
The setup shown here is a very simple one compared to the methods popular
software like *BLAST* use.
Since the *k-mer* matching step is very fast and the gapped alignments take the
largest part of the time, you usually want to have additional filters before
you trigger a gapped alignment:
For example, commonly a gapped alignment is only started at a match, if there is
another match on the same diagonal in proximity.
Furthermore, the parameter selection, e.g. the *k-mer* length, is key to a fast
but also sensitive alignment procedure.
However, you can find suitable parameters in literature or run benchmarks by
yourself to find appropriate parameters for your application.