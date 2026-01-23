.. include:: /tutorial/preamble.rst

Structural alphabets
====================

.. currentmodule:: biotite.structure.alphabet

In the previous chapters we have seen the multitude of methods, that can be applied
to sequence and structure data.
*Structural alphabets* combine the best of both worlds:
A structural alphabet is a representation of protein or nucleic acid structures,
where each residue is encoded into a single character, based on the local geometry
or contact partners of that residue.
This way the high performance sequence-based methods, for example alignment searches,
can be applied to structure data.

The :mod:`biotite.structure.alphabet` subpackage provides multiple structural alphabets,
but for the scope of this tutorial we will focus on the *3Di* alphabet, popularized by
the `Foldseek <https://github.com/steineggerlab/foldseek>`_ software for fast protein
structure comparison.
Just keep in mind that in the following examples the underlying structural alphabet
can be substituted with minimal modifications.

Converting structures to sequences
----------------------------------
We start by getting the structure of our protein of interest.
In this case we will use ferredoxin from *E. coli* (PDB ID: ``2ZVS``).
After filtering out all non-amino acid residues, we create the *3Di* sequence for each
chain with :func:`to_3di()`.

.. jupyter-execute::

    import biotite.database.rcsb as rcsb
    import biotite.structure as struc
    import biotite.structure.alphabet as strucalph
    import biotite.structure.io.pdbx as pdbx

    # You can actually directly read the downloaded PDBx file content
    # without an intermediate file
    pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch("2ZVS", "bcif"))
    ec_ferredoxin = pdbx.get_structure(pdbx_file, model=1)
    # This structural alphabet expects a peptide chain
    ec_ferredoxin = ec_ferredoxin[struc.filter_amino_acids(ec_ferredoxin)]
    structural_sequences, chain_starts = strucalph.to_3di(ec_ferredoxin)
    print(structural_sequences)
    print(chain_starts)

Note that :func:`to_3di()` returns not a single :class:`I3DSequence` sequence but a list
of sequences, one for each chain in the structure.
Accompanying the sequences, the function also returns the atom indices where each of the
chains starts.
As our structure contains only one chain, the desired 3Di sequence is the first and only
element in the list.

.. jupyter-execute::

    ec_ferredoxin_3di = structural_sequences[0]
    print(ec_ferredoxin_3di)

Each symbol in this rather cryptic sequence corresponds to one residue in the structure.
To get the corresponding residues as :class:`.AtomArray` objects we can use the
residue-level functionality of :mod:`biotite.structure`.
While the sequence is hardly human-readable, its true power lies in its ability to
be compared to *3Di* sequences from other proteins.

Sequence alignments on structural alphabets
-------------------------------------------
As mentioned in the :doc:`sequence chapter <../sequence/encoding>`, the sequence
based methods in :mod:`biotite.sequence` generally do not care about the type of
sequence.
This means that we can use any method we have learned so far on structural sequences as
well.
For the scope of this tutorial we will merely use :func:`align_optimal()` to find
corresponding residues in two structures.
As the structure is generally much better conserved than its sequence, the alignment of
*3Di* sequences will even work on remote homologs with low amino acid sequence identity,
where a classical sequence alignment would fail.
To demonstrate this, we will compare the *E. coli* ferredoxin with the remotely similar
ferredoxin from the thermophilic archaeon *S. tokodaii*.

.. jupyter-execute::

    pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch("1XER", "bcif"))
    st_ferredoxin = pdbx.get_structure(pdbx_file, model=1)
    st_ferredoxin = st_ferredoxin[struc.filter_amino_acids(st_ferredoxin)]
    st_ferredoxin_3di = strucalph.to_3di(st_ferredoxin)[0][0]

To align the two 3Di sequences, we merely need a :class:`.SubstitutionMatrix` that
matches the alphabet of the :class:`I3DSequence`.
Like for amino acid and nucleotide sequences, :mod:`biotite.sequence.align` provides
it out of the box with :func:`.SubstitutionMatrix.std_3di_matrix()`.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import biotite.sequence.align as align
    import biotite.sequence.graphics as graphics

    matrix = align.SubstitutionMatrix.std_3di_matrix()
    alignment = align.align_optimal(
        ec_ferredoxin_3di,
        st_ferredoxin_3di,
        matrix,
        gap_penalty=(-10, -1),
        terminal_penalty=False,
    )[0]

    fig, ax = plt.subplots(figsize=(8.0, 2.0))
    graphics.plot_alignment_similarity_based(
        ax, alignment, matrix=matrix, labels=["EC", "ST"], symbols_per_line=50
    )

If you prefer coloring the symbols in the alignment by their type, you are lucky:
:mod:`biotite.sequence.graphics` provides a
:doc:`color scheme <../../examples/gallery/sequence/misc/color_schemes>` for each of the
supported structural alphabets as well.

.. jupyter-execute::

    fig, ax = plt.subplots(figsize=(8.0, 2.0))
    graphics.plot_alignment_type_based(
        ax, alignment, labels=["EC", "ST"], symbols_per_line=50
    )

Example: Superimposing structures
---------------------------------
One typical use case of structural alphabets is superimposing structures of remote
homologs.
Here the challenge is finding the corresponding residues in the two structures, whose
squared distance the superimposition algorithm should minimize.
The solution is to use the alignment of the structural alphabet:
One simply inputs the ``CA`` atoms of the aligned residues.

.. jupyter-execute::

    def rmsd_from_alignment(fixed, mobile, alignment):
        """
        A very simple function that extracts corresponding residues (the 'anchors')
        from an alignment and uses them to run a superimposition.
        Finally the RMSD of the superimposed structures plus the number of anchors is
        returned.
        """
        alignment_codes = align.get_codes(alignment)
        anchor_mask = (
            # Anchors must be structurally similar
            (matrix.score_matrix()[alignment_codes[0], alignment_codes[1]] > 0)
            # Gaps are not anchors
            & (alignment_codes != -1).all(axis=0)
        )
        superimposition_anchors = alignment.trace[anchor_mask]
        # Each anchor corresponds to a residue
        # Use the CA atoms as representative for each residue
        fixed_ca = fixed[fixed.atom_name == "CA"]
        mobile_ca = mobile[mobile.atom_name == "CA"]
        fixed_anchors = fixed_ca[superimposition_anchors[:, 0]]
        mobile_anchors = mobile_ca[superimposition_anchors[:, 1]]

        mobile_anchors, transform = struc.superimpose(
            fixed_anchors,
            mobile_anchors,
        )
        return struc.rmsd(fixed_anchors, mobile_anchors), len(superimposition_anchors)

    rmsd, n_anchors = rmsd_from_alignment(ec_ferredoxin, st_ferredoxin, alignment)
    print("Number of corresponding residues found:", n_anchors)
    print("RMSD:", rmsd)

Again, with a classical amino acid sequence based approach the accuracy of the
superimposition would be much lower:
In this case less corresponding residues can be found from the amino sequence
alignment and the RMSD between them is significantly higher.

.. jupyter-execute::

    matrix = align.SubstitutionMatrix.std_protein_matrix()
    ec_ferredoxin_seq = struc.to_sequence(ec_ferredoxin)[0][0]
    st_ferredoxin_seq = struc.to_sequence(st_ferredoxin)[0][0]
    alignment = align.align_optimal(
        ec_ferredoxin_seq,
        st_ferredoxin_seq,
        matrix,
        gap_penalty=(-10, -1),
        terminal_penalty=False,
    )[0]
    rmsd, n_anchors = rmsd_from_alignment(ec_ferredoxin, st_ferredoxin, alignment)
    print("Number of corresponding residues found:", n_anchors)
    print("RMSD:", rmsd)

    fig, ax = plt.subplots(figsize=(8.0, 2.0))
    graphics.plot_alignment_similarity_based(
        ax, alignment, matrix=matrix, labels=["EC", "ST"], symbols_per_line=50
    )

This shows only a small fraction of the versatility of structural alphabets.
They can also be used to find structural homologs in a large database, to superimpose
multiple structures at once and much more!