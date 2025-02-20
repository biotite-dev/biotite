.. include:: /tutorial/preamble.rst

Interface to RDKit
==================

.. currentmodule:: biotite.interface.rdkit

`RDKit <https://www.rdkit.org/>`_ is a popular cheminformatics package
and thus can be used to supplement *Biotite* with a variety of functionalities focused
on small molecules, such as conversion from/to textual representations
(e.g. *SMILES* and *InChI*) and visualization as structural formulas.
Basically, the :mod:`biotite.interface.rdkit` subpackage provides only two functions:
:func:`to_mol()` to obtain a :class:`rdkit.Chem.rdchem.Mol` from an :class:`.AtomArray`
and :func:`from_mol()` for the reverse direction.
The rest happens within the realm of *RDKit*.
This tutorial will only give a small glance on how the interface can be used.
For comprehensive documentation refer to the
`RDKit documentation <https://www.rdkit.org/docs/>`_.

First example: Depiction as structural formula
----------------------------------------------
*RDKit* allows rendering structural formulas using
`pillow <https://pillow.readthedocs.io/en/stable/>`_.
For a proper structural formula, we need to compute proper 2D coordinates first.

.. jupyter-execute::

    import biotite.interface.rdkit as rdkit_interface
    import biotite.structure.info as struc
    import rdkit.Chem.AllChem as Chem
    from rdkit.Chem.Draw import MolToImage

    penicillin = struc.residue("PNN")
    mol = rdkit_interface.to_mol(penicillin)
    # We do not want to include explicit hydrogen atoms in the structural formula
    mol = Chem.RemoveHs(mol)
    Chem.Compute2DCoords(mol)
    image = MolToImage(mol, size=(600, 400))
    display(image)

Second example: Creating a molecule from SMILES
-----------------------------------------------
Although the *Chemical Component Dictionary* accessible from
:mod:`biotite.structure.info` already provides all compounds found in the PDB,
there are a myriad of compounds out there that are not part of it.
One way to to obtain them as :class:`.AtomArray` is passing a *SMILES* string to
*RDKit* to obtain the topology of the molecule and then computing the coordinates.

.. jupyter-execute::

    ERTAPENEM_SMILES = "C[C@@H]1[C@@H]2[C@H](C(=O)N2C(=C1S[C@H]3C[C@H](NC3)C(=O)NC4=CC=CC(=C4)C(=O)O)C(=O)O)[C@@H](C)O"

    mol = Chem.MolFromSmiles(ERTAPENEM_SMILES)
    # RDKit uses implicit hydrogen atoms by default, but Biotite requires explicit ones
    mol = Chem.AddHs(mol)
    # Create a 3D conformer
    conformer_id = Chem.EmbedMolecule(mol)
    Chem.UFFOptimizeMolecule(mol)
    ertapenem = rdkit_interface.from_mol(mol, conformer_id)
    print(ertapenem)