# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("as_mask", [False, True])
def test_find_connected(seed, as_mask):
    """
    The ``label_asym_id`` in PDBx files distinguishes different molecules.
    Therefore, all connected atoms should hame the same chain ID.
    An exception are water molecules, which have the same chain ID, albeit not being
    bonded to each other.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), "1k6p.bcif"))
    atoms = pdbx.get_structure(
        pdbx_file, model=1, include_bonds=True, use_author_fields=False
    )
    atoms = atoms[~struc.filter_solvent(atoms)]

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    root = rng.integers(atoms.array_length())
    chain_id = atoms.chain_id[root]

    connected_indices = struc.find_connected(atoms.bonds, root, as_mask=as_mask)
    assert np.all(atoms.chain_id[connected_indices] == chain_id)
    if as_mask:
        # When we have a boolean mask, we can also check
        # if all non-connected atoms have a different chain ID
        assert np.all(atoms.chain_id[~connected_indices] != chain_id)


@pytest.mark.parametrize(
    "res_name, expected_bonds",
    [
        # Easy ligand visualization at:
        # https://www.rcsb.org/ligand/<ABC>
        ("TYR", [
            ("N",   "CA" ),
            ("CA",  "C"  ),
            ("CA",  "CB" ),
            ("C",   "OXT"),
            ("CB",  "CG" ),
            ("CZ",  "OH" ),
        ]),
        ("CEL", [
            ("C1",   "C4" ),
            ("C8",   "C11"),
            ("C15",  "S1" ),
            ("N3",   "S1" )
        ]),
        ("LEO", [
            ("C3",   "C8" ),
            ("C6",   "C17"),
            ("C17",  "C22"),
        ]),
    ]
)  # fmt: skip
def test_find_rotatable_bonds(res_name, expected_bonds):
    """
    Check the :func:`find_rotatable_bonds()` function based on
    known examples.
    """
    molecule = info.residue(res_name)

    ref_bond_set = {
        tuple(sorted((name_i, name_j))) for name_i, name_j in expected_bonds
    }

    rotatable_bonds = struc.find_rotatable_bonds(molecule.bonds)
    test_bond_set = set()
    for i, j, _ in rotatable_bonds.as_array():
        test_bond_set.add(tuple(sorted((molecule.atom_name[i], molecule.atom_name[j]))))

    # Compare with reference bonded atom names
    assert test_bond_set == ref_bond_set
    # All rotatable bonds must be single bonds
    assert np.all(rotatable_bonds.as_array()[:, 2] == struc.BondType.SINGLE)


@pytest.mark.parametrize(
    "cif_path, expected_bond_indices",
    [
        (
            join(data_dir("structure"), "3o5r.cif"),
            [252, 257],  # Carbonyl carbon and subsequent backbone nitrogen
        )
    ],
)
def test_canonical_bonds_with_altloc_occupancy(cif_path, expected_bond_indices):
    """
    Test whether canonical inter-residue bonds are correctly computed when
    `altloc="occupancy"` and the higher-occupancy atom occurs second in the CIF file.
    """

    cif_file = pdbx.CIFFile.read(cif_path)
    atom_array = pdbx.get_structure(
        cif_file.block, altloc="occupancy", include_bonds=True
    )

    atom1, atom2 = expected_bond_indices

    # Assert that the canonical inter-residue bond exists
    assert atom2 in atom_array.bonds.get_bonds(atom1)[0]


@pytest.mark.parametrize("periodic", [False, True])
def test_method_consistency(periodic):
    """
    Check if :func:`connect_via_distances()` and
    :func:`connect_via_residue_names()` give the same bond list
    """
    THRESHOLD_PERCENTAGE = 0.99

    # Structure with peptide, nucleotide, small molecules and water
    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), "5ugo.bcif"))
    atoms = pdbx.get_structure(pdbx_file, model=1)
    if periodic:
        # Add large dummy box to test parameter
        # No actual bonds over the periodic boundary are expected
        atoms.box = np.identity(3) * 100

    bonds_from_names = struc.connect_via_residue_names(atoms)
    bonds_from_names.remove_bond_order()

    bonds_from_distances = struc.connect_via_distances(atoms, periodic=periodic)

    # The distance based method may not detect all bonds
    assert bonds_from_distances.as_set().issubset(bonds_from_names.as_set())
    assert (
        len(bonds_from_distances.as_array())
        >= len(bonds_from_names.as_array()) * THRESHOLD_PERCENTAGE
    )
