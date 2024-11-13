.. include:: /tutorial/preamble.rst

Profiles and position-specific scoring matrices
===============================================
Often sequences are not viewed in isolation:
For example, if you investigate a protein family, you do not handle a single sequence,
but an arbitrarily large collection of highly similar sequences.
At some point handling those sequences as a mere collection of individual sequences
becomes impractical for answering common questions, like which are the prevalent
amino acids at a certain positions of the sequences.

Sequence profiles
-----------------

.. currentmodule:: biotite.sequence

This is where *sequence profiles* come into play:
The condense the a collection of aligned sequences into a matrix that tracks the
frequency of each symbol at each position of the alignment.
Hence, asking the questions such as '*How frequent is an alanine at the n-th position?*'
becomes a trivial indexing operation.

As example for profiles, we will reuse the cyclotide sequence family from the
:doc:`previous chapter <align_multiple>`.

.. jupyter-execute::

    from tempfile import NamedTemporaryFile
    import biotite.sequence.io.fasta as fasta
    import biotite.database.entrez as entrez

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
    # Extract cyclotide N as query sequence for later
    query = sequences.pop("N")

To create a profile, we first need to align the sequences, so corresponding symbols
appear the same position.
Then we can create a :class:`SequenceProfile` object from the :class:`.Alignment`, which
simply counts for each alignment column (i.e. the sequence position) the number of
occurrences for each symbol.

.. jupyter-execute::

    import biotite.sequence as seq
    import biotite.sequence.align as align

    alignment, _, _, _ = seq.align.align_multiple(
        list(sequences.values()),
        align.SubstitutionMatrix.std_protein_matrix(),
        gap_penalty=-5,
    )
    profile = seq.SequenceProfile.from_alignment(alignment)
    print(profile)

Each row in the displayed count matrix
(accessible via :attr:`SequenceProfile.symbols`) refers to a single position, i.e. a
column in the input MSA, and each column refers to a symbol in the underlying alphabet
(accessible via :attr:`SequenceProfile.alphabet`).
For completeness it should be noted that :attr:`SequenceProfile.gaps` also tracks the
gaps for each position in the alignment, but we will not further use them in this
tutorial.

Note that the information about the individual sequences is lost in the condensation
process: There is no way to reconstruct the original sequences from the profile.
However, we can still extract a consensus sequence from the profile, which is a
sequence that represents the most frequent symbol at each position.

.. jupyter-execute::

    print(profile.to_consensus())

Profile visualization as sequence logo
--------------------------------------

.. currentmodule:: biotite.sequence.align

A common way to visualize a sequence profile is a sequence logo.
It depicts each profile position as a stack of letters:
The degree of conversation (more precisely the
`Shannon entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_)
is the height of a stack and each letter's height in the stack is proportional to its
frequency at the respective position.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    from biotite.sequence.graphics import plot_sequence_logo

    fig, ax = plt.subplots(figsize=(8.0, 2.0), constrained_layout=True)
    plot_sequence_logo(ax, profile)
    ax.set_xlabel("Residue position")
    ax.set_ylabel("Bits")

Position-specific scoring matrices
----------------------------------

.. currentmodule:: biotite.sequence.align

Sequence profiles can achieve even more:
The substitution matrices we have seen so far assign a score to a pair of two symbols,
regardless of their position in a sequence.
However, as discussed above, the position is crucial information to determine how
conserved a certain symbol is in a family of sequences.

Hence, we extend the concept of substitution matrices to *position-specific scoring
matrices*,
which assign a score to a symbol and a position (instead of another symbols).
A typical way to create a position-specific scoring matrix is to use the log-odds matrix
from a profile.

.. jupyter-execute::

    # For a real analysis amino acid background frequencies should be provided
    log_odds = profile.log_odds_matrix(pseudocount=1)

Now, we encounter a problem:
To create a :class:`SubstitutionMatrix` object from the log-odds matrix, we require two
:class:`.Alphabet` objects:
One is taken from the query sequence to be aligned to the profile, but which alphabet
do we use for the positional axis of the matrix?
Likewise, the alignment functions (e.g. :func:`align_optimal()`) require a two sequences
to be aligned, but we only have one query sequence.

To solve this problem :mod:`biotite.sequence` provides the :class:`.PositionalSequence`
which acts as a placeholder for the second sequence in the alignment.
Its alphabet contains a unique symbol for each position, i.e. the alphabet has the
sought length.

.. jupyter-execute::

    positional_seq = seq.PositionalSequence(profile.to_consensus())
    matrix = align.SubstitutionMatrix(
        positional_seq.alphabet,
        seq.ProteinSequence.alphabet,
        # Substitution matrices require integer scores
        # Multiply by 10 to increase value range and convert to integer
        (log_odds * 10).astype(int)
    )
    alignment = align.align_optimal(
        positional_seq, query, matrix, gap_penalty=-5, max_number=1
    )[0]
    print(alignment)

Only the length of the input sequence passed to :class:`.PositionalSequence` matters for
the alignment to work.
The consensus sequence of the profile was merely chosen for cosmetic reasons, i.e.
to have a meaningful string representation of the positional sequence and thus the
alignment.

Further applications
^^^^^^^^^^^^^^^^^^^^

Using a sequence profile is only one way to create a position-specific scoring matrix.
The combination of :class:`.PositionalSequence` objects and a corresponding
:class:`.SubstitutionMatrix` allows to assign a score between two sequence positions
based on any property.

As a toy example, let's say we want to harness the information that cyclotides have
strongly conserved disulfide bridges.
Hence, we want the score matrix not only to reward when cysteines are paired with
cysteines in general, but specifically when cysteines are paired with cysteines at the
corresponding position, i.e. the first cysteine with the first cysteine etc.
As template we will use the standard *BLOSUM62* substitution matrix, expanded into
a position-specific scoring matrix using :meth:`SubstitutionMatrix.as_positional()`:

.. jupyter-execute::

    cyclotide_n = query
    cyclotide_c = sequences["C"]
    aa_matrix = align.SubstitutionMatrix.std_protein_matrix()
    # `as_positional()` expands the matrix into a position-specific scoring matrix
    # for the given sequences while also giving us the positional sequences for them
    pos_matrix, pos_cyclotide_n, pos_cyclotide_c = aa_matrix.as_positional(
        cyclotide_n, cyclotide_c
    )
    # Introduce bias for cysteine positions
    score_matrix = pos_matrix.score_matrix().copy()
    cys_code = seq.ProteinSequence.alphabet.encode("C")
    score_matrix[cyclotide_n.code == cys_code, cyclotide_c.code == cys_code] = 100
    disulfide_matrix = align.SubstitutionMatrix(
        pos_matrix.get_alphabet1(),
        pos_matrix.get_alphabet2(),
        score_matrix,
    )
    print(disulfide_matrix)

You might notice the new shape of the substitution matrix:
Instead of spanning the 24 x 24 amino acids including ambiguous symbols, it now matches
the lengths of the two input sequences.
In this matrix you can spot the biased scores at the matching cysteine positions.