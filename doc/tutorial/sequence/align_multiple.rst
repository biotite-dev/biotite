.. include:: /tutorial/preamble.rst

Guide trees and multiple sequence alignments
============================================

.. currentmodule:: biotite.sequence.align

In :doc:`a previous chapter <align_optimal>` we have learned how to align *two*
sequences with each other.
However, for many use cases we require an alignment of more than two
sequences.
For example, one such use case is the analysis of homologous regions within a
protein family.
Although the *dynamic programming* algorithm behind :func:`align_optimal()` can in
theory be extended to any number of sequences, the computation time scales
linearly with the length *each* aligned sequence.
Thus, the method becomes infeasible for already a few sequences.

To accelerate the computation of pairwise sequence alignments we have seen
:doc:`heuristic methods <align_heuristic>`, that increase the computation speed
drastically, but may not always find the optimal solution (but often still do).
For *multiple sequence alignments* (MSAs) we can also resort to heuristics to
find good solutions in reasonable time, the so called *progressive alignment*:
Instead of finding the optimal alignment of all sequences at once, we compute
a number of pairwise alignments and combine these alignments to eventually
get an alignment that contains all sequences: the MSA.
Therefore, the method needs some *guide tree* to know in which order to align
the sequences.

In this chapter we will first learn how to construct guide trees (and trees
in general) and then use them to inform the MSA algorithm.

Phylogenetic and guide trees
----------------------------

.. currentmodule:: biotite.sequence.phylo

Trees have an important role in bioinformatics, as they are used to
guide multiple sequence alignments or to create phylogenies.

In *Biotite* such a tree is represented by the :class:`Tree` class in
the :mod:`biotite.sequence.phylo` package.
Each node in a tree has a number of child nodes (none in case of leaf nodes)
but only one parent node.
The exception is the *root node*, which has no parent.
Each node in a tree is represented by a :class:`TreeNode`.
When a :class:`TreeNode` is created, you have to provide either child
nodes and their distances to this node (intermediate node) or a
reference index (leaf node).
This reference index is dependent on the context and can refer to
anything: sequences, organisms, etc.

The children and the reference index cannot be changed after object
creation.
Also the parent can only be set once - when the node is used as child
in the creation of a new node.

.. jupyter-execute::

    import biotite.sequence.phylo as phylo

    # The reference objects
    fruits = ["Apple", "Pear", "Orange", "Lemon", "Banana"]
    # Create nodes
    apple  = phylo.TreeNode(index=fruits.index("Apple"))
    pear   = phylo.TreeNode(index=fruits.index("Pear"))
    orange = phylo.TreeNode(index=fruits.index("Orange"))
    lemon  = phylo.TreeNode(index=fruits.index("Lemon"))
    banana = phylo.TreeNode(index=fruits.index("Banana"))
    intermediate1 = phylo.TreeNode(
        children=(apple, pear), distances=(2.0, 2.0)
    )
    intermediate2 = phylo.TreeNode((orange, lemon), (1.0, 1.0))
    intermediate3 = phylo.TreeNode((intermediate2, banana), (2.0, 3.0))
    root = phylo.TreeNode((intermediate1, intermediate3), (2.0, 1.0))
    # Create tree from root node
    tree = phylo.Tree(root=root)
    # Trees can be converted into Newick notation
    print("Tree:", tree.to_newick(labels=fruits))
    # Distances can be omitted
    print(
        "Tree w/o distances:",
        tree.to_newick(labels=fruits, include_distance=False)
    )
    # Distances can be measured
    distance = tree.get_distance(fruits.index("Apple"), fruits.index("Banana"))
    print("Distance Apple-Banana:", distance)

You can plot a tree as dendrogram.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import biotite.sequence.graphics as graphics

    fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)
    graphics.plot_dendrogram(ax, tree, labels=fruits)

From distances to trees
^^^^^^^^^^^^^^^^^^^^^^^
In most scenarios we do not have a tree from the beginning, but only
distances between the nodes, e.g. distances from an alignment.
To create a :class:`Tree` from these distances, we can use
*hierarchical clustering* methods provided by :mod:`biotite.sequence.phylo`,
namely :func:`upgma()` and :func:`neighbour_joining()`.

.. jupyter-execute::

    import numpy as np

    distances = np.array([
        [ 0, 17, 21, 31, 23],
        [17, 0,  30, 34, 21],
        [21, 30, 0,  28, 39],
        [31, 34, 28,  0, 43],
        [23, 21, 39, 43,  0]
    ])
    tree = phylo.upgma(distances)
    fig, ax = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    graphics.plot_dendrogram(ax, tree, orientation="top")

Multiple sequence alignments
----------------------------

.. currentmodule:: biotite.sequence.align

Now that we know how guide trees work, we proceed to MSAs.
:mod:`biotite.sequence.align` implements an progressive alignment method in
:func:`align_multiple()`.
First we will fetch some homologous sequences from the NCBI database.

.. jupyter-execute::

    from tempfile import NamedTemporaryFile

    import biotite.sequence.align as align
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
    sequences = {
        # The cyclotide variant is the last character in the header
        header[-1]: seq for header, seq in fasta.get_sequences(fasta_file).items()
    }
    for name, sequence in sequences.items():
        print(f"{name}: {sequence}")

By default we do not have to provide a guide tree at all:
The function will compute pairwise alignments and construct a guide tree from
distances between these alignments.

.. jupyter-execute::

    alignment, order, guide_tree, distance_matrix = align.align_multiple(
        list(sequences.values()),
        matrix=align.SubstitutionMatrix.std_protein_matrix(),
        gap_penalty=-5,
    )

    # Order alignment according to guide tree
    alignment = alignment[:, order.tolist()]
    labels = np.array(list(sequences.keys()))[order]
    fig, ax = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    graphics.plot_alignment_type_based(
        ax, alignment, color_scheme="blossom",
        symbols_per_line=len(alignment), symbol_size=10,
        labels=labels, label_size=12,
    )

We can inspect the guide tree that was used to create the alignment

.. jupyter-execute::

    fig, ax = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    graphics.plot_dendrogram(
        ax, guide_tree, orientation="top",
        labels=list(sequences.keys()), show_distance=False
    )
    _ = ax.set_yticks([])

Custom guide trees
^^^^^^^^^^^^^^^^^^
:func:`align_multiple()` also accepts a custom guide tree, for example in
case we would like to use a different distance metric or another hierarchical
clustering method.
For demonstration purposes we will create a distance matrix from the number
of shared *k-mers* between the sequences
(we already encountered *k-mers* in the
:doc:`previous chapter <align_heuristic>`).
This is actually a commonly used method in prominent MSA software such as
*MUSCLE* and *MAFFT*, because counting *k-mer* matches is much faster than
computing pairwise alignments.

.. jupyter-execute::

    K = 3

    kmer_table = align.KmerTable.from_sequences(
        k=K, sequences=list(sequences.values())
    )
    match_number_matrix = np.zeros((len(sequences), len(sequences)), dtype=int)
    for i, sequence in enumerate(sequences.values()):
        # Avoid that multiple k-mers match to the same position on other sequence
        kmers = kmer_table.kmer_alphabet.create_kmers(sequence.code)
        unique_kmers, positions = np.unique(kmers, return_counts=True)
        matches = kmer_table.match_kmer_selection(positions, unique_kmers)
        matched_seq_indices = matches[:, 1]
        # Count how many times this sequence matches with each other sequence
        match_number = np.bincount(matched_seq_indices, minlength=len(sequences))
        match_number_matrix[i] = match_number
    print("Number of k-mer matches:")
    print(match_number_matrix, end="\n\n")

    # Use very simplified distance formula adapted from the MUSCLE method
    sequence_lengths = np.array([len(seq) for seq in sequences.values()])
    min_seq_length_matrix = np.min(
        np.stack([
            np.repeat(sequence_lengths, len(sequences)).reshape(match_number_matrix.shape),
            np.tile(sequence_lengths, len(sequences)).reshape(match_number_matrix.shape)
        ], axis=-1),
        axis=-1
    )
    # If a sequence contains duplicates of a k-mer,
    # match_number_matrix[i,j] may differ slightly from match_number_matrix[j,i]
    # Simple measure to make the matrix symmetric
    match_number_matrix = (match_number_matrix + match_number_matrix.T) / 2
    fractional_similarity = match_number_matrix / (min_seq_length_matrix - K + 1)
    distances = 1 - fractional_similarity
    print("Distance matrix:")
    print(distances)

Now we will use :func:`upgma()` to compute the guide tree for the MSA from the
custom distance matrix.

.. jupyter-execute::

    guide_tree = phylo.upgma(distances)
    fig, ax = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    graphics.plot_dendrogram(
        ax, guide_tree, orientation="top",
        labels=list(sequences.keys()), show_distance=False
    )
    _ = ax.set_yticks([])

When you compare this guide tree with the one from the previous MSA, you will
note some small differences.
However, considering we potentially saved a lot of computation time, overall
the trees are quite similar.
Finally we can use this guide tree as input to :func:`align_multiple()`.

.. jupyter-execute::

    # Same procedure as above, but now with our custom guide tree
    alignment, order, _, _ = align.align_multiple(
        list(sequences.values()),
        matrix=align.SubstitutionMatrix.std_protein_matrix(),
        gap_penalty=-5,
        guide_tree=guide_tree
    )
    alignment = alignment[:, order.tolist()]
    labels = np.array(list(sequences.keys()))[order]
    fig, ax = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    graphics.plot_alignment_type_based(
        ax, alignment, color_scheme="blossom",
        symbols_per_line=len(alignment), symbol_size=10,
        labels=labels, label_size=12,
    )

Other options
^^^^^^^^^^^^^
:func:`align_multiple()` is only recommended for strongly related sequences or
exotic sequence types.
When high accuracy or computation time matters, other MSA programs deliver
better results.
In a :doc:`later chapter <../application/msa>` we will see how to use these MSA
programs in an easy and seamless way.