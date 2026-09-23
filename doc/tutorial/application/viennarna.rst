.. include:: /tutorial/preamble.rst

RNA secondary structure prediction
==================================

.. currentmodule:: biotite.application.viennarna

The *ViennaRNA* package is a collection of programs for the prediction and
analysis of RNA secondary structures.
Each of them is interfaced by a separate class in the
:mod:`biotite.application.viennarna` subpackage.
The most prominent one is ``RNAfold``, that predicts the secondary structure
with the minimum free energy for a given sequence.
Its interface is the :class:`FoldApp`.
As example we take the famous tRNA-Phe from yeast, which forms the cloverleaf
structure found in tRNAs.

.. jupyter-execute::

    import biotite.sequence as seq
    import biotite.application.viennarna as viennarna

    # Note that RNA sequences use 'T' instead of 'U' in Biotite
    trna = seq.NucleotideSequence(
        "GCGGATTTAGCTCAGTTGGGAGAGCGCCAGACTGAAGATCTGG"
        "AGGTCCTGTGTTCGATCCACAGAATTCGCACCA"
    )
    app = viennarna.FoldApp()
    folded = app.run(trna).result()
    print(folded.dot_bracket)
    print(folded.free_energy)

The result is a :class:`FoldedRNA`, that describes the structure in
*dot-bracket notation*:
Each pair of matching brackets is a base pair and each dot is an unpaired
base.
The four stems of the cloverleaf are clearly visible.
For further processing, the structure is also available as an array of base
pairs, giving the indices of the paired bases in the sequence.

.. jupyter-execute::

    base_pairs = folded.base_pairs()
    print(base_pairs[:5])

The energy parameters of the prediction depend on the temperature.
Hence, by increasing the temperature the structure gets less stable, until it
eventually melts.

.. jupyter-execute::

    for temperature in [37, 60, 80]:
        folded_at_temp = app.run(trna, temperature=temperature).result()
        print(f"{temperature} °C: {folded_at_temp.dot_bracket}")
        print(f"       {folded_at_temp.free_energy:6.2f} kcal/mol")

If some base pairs or unpaired bases are already known, e.g. from experiments,
they can be given as constraints via the ``pairs``, ``paired`` and
``unpaired`` parameters to guide the prediction.

Plotting the structure
----------------------
The dot-bracket notation is not very intuitive to read.
Fortunately, *ViennaRNA* also provides ``RNAplot``, that computes 2D
coordinates for each base, so that the structure can be drawn as a
graph.
The corresponding :class:`PlotApp` returns these coordinates as array, that
can be plotted with *Matplotlib*.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    coord = viennarna.PlotApp().run(folded.dot_bracket).result()

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    # Connect consecutive bases along the backbone
    ax.plot(*coord.T, color="black", linewidth=1, zorder=1)
    # Connect paired bases
    ax.add_collection(
        LineCollection(
            [(coord[i], coord[j]) for i, j in base_pairs],
            color="silver",
            linewidth=1,
            zorder=1,
        )
    )
    # Draw each base as circle with its symbol, using 'U' for RNA
    ax.scatter(*coord.T, s=140, color="white", edgecolor="black", zorder=2)
    for (x, y), symbol in zip(coord, str(trna).replace("T", "U")):
        ax.text(x, y, symbol, ha="center", va="center", fontsize=8, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

Beyond the minimum free energy structure
----------------------------------------
The minimum free energy structure is only the most stable one among many
possible structures.
``RNAsubopt``, interfaced by the :class:`SuboptApp`, enumerates all
structures within a given energy range above the minimum.

.. jupyter-execute::

    suboptimal = viennarna.SuboptApp().run(trna, energy_range=0.5).result()
    for folded_alternative in sorted(
        suboptimal, key=lambda folded: folded.free_energy
    ):
        print(folded_alternative.dot_bracket, folded_alternative.free_energy)

The other classes of the subpackage cover further programs from *ViennaRNA*,
for example :class:`CofoldApp` for the structure of two hybridized strands,
:class:`AlifoldApp` for the consensus structure of an alignment or
:class:`PKplexApp` for structures containing pseudoknots.
