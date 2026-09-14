.. include:: /tutorial/preamble.rst

Molecular docking
=================

.. currentmodule:: biotite.application_v2.autodock

*AutoDock Vina* predicts how a small molecule, the *ligand*, binds to a
protein, the *receptor*.
It is interfaced by the :class:`VinaApp` in the
:mod:`biotite.application_v2.autodock` subpackage.
Let's dock biotin into its famous binding partner streptavidin.
As we would like to check how well the docking works, we download a
high-resolution crystal structure of the complex, which we split into the
receptor and the reference binding pose of the ligand.
The docking requires bonds and formal charges, hence both are included when
loading the structure.

.. jupyter-execute::

    from tempfile import gettempdir
    import biotite.database.rcsb as rcsb
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    pdbx_file = pdbx.BinaryCIFFile.read(
        rcsb.fetch("2rtg", "bcif", gettempdir())
    )
    structure = pdbx.get_structure(
        pdbx_file, model=1, include_bonds=True, extra_fields=["charge"]
    )
    # The structure is a homodimer, one monomer is sufficient
    structure = structure[structure.chain_id == "B"]
    receptor = structure[struc.filter_amino_acids(structure)]
    ref_ligand = structure[structure.res_name == "BTN"]

The ligand should not be taken from the complex, as it is already in its
bound conformation.
Instead we take the idealized model of biotin from the *Chemical Component
Dictionary*, that comes with the required bonds and charges.

.. jupyter-execute::

    import biotite.structure.info as info

    ligand = info.residue("BTN")

Now we let *Vina* place the ligand into the receptor.
The search space is a box, defined by its center and size.
Here we cheat a little bit by centering the box at the position of the
reference ligand, which we would not know for an unknown complex.

.. jupyter-execute::

    import biotite.application_v2.autodock as autodock

    app = autodock.VinaApp()
    result = app.run(
        ligand,
        receptor,
        center=struc.centroid(ref_ligand),
        # Edge length of the box in Å
        size=[20, 20, 20],
        # A fixed seed and a single thread make the run reproducible
        seed=42,
        cpu=1,
    ).result()
    print(result.energies)

*Vina* reports multiple *binding modes*, sorted from best to worst predicted
binding energy.
The coordinates of the ligand in each binding mode are given by
:attr:`VinaResult.ligand_coord`, which we combine with the input ligand to an
:class:`AtomArrayStack`.
Note that *Vina* removes nonpolar hydrogen atoms, so their coordinates are
*NaN*.
As we compare only the heavy atoms with the reference pose anyway, we simply
remove all hydrogen atoms.

.. jupyter-execute::

    import numpy as np

    docked_ligand = struc.from_template(ligand, result.ligand_coord)
    docked_ligand = docked_ligand[..., struc.filter_heavy(docked_ligand)]
    ref_ligand = ref_ligand[struc.filter_heavy(ref_ligand)]
    # Ensure that both models have the same atom order
    docked_ligand = docked_ligand[..., info.standardize_order(docked_ligand)]
    ref_ligand = ref_ligand[info.standardize_order(ref_ligand)]
    # No superimposition, as the position in the binding pocket matters
    rmsd = struc.rmsd(ref_ligand, docked_ligand)
    for energy, deviation in zip(result.energies, rmsd):
        print(f"{energy:6.2f} kcal/mol  {deviation:5.2f} Å")

The binding mode with the best predicted energy is also the one that is
closest to the experimentally determined pose.
Let's look at it visually:

.. jupyter-execute::

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(result.energies, rmsd, marker="+", color="black")
    ax.set_xlabel("Binding energy (kcal/mol)")
    ax.set_ylabel("RMSD to reference (Å)")
    fig.tight_layout()

Vina can also treat selected side chains of the receptor as flexible during
the docking.
For this purpose, a boolean mask of the flexible residues is given to the
``flexible`` parameter of :meth:`VinaApp.run()`.
