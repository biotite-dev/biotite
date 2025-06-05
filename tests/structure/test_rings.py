import itertools
from enum import IntEnum
from pathlib import Path
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.fixture
def riboswitch_structure():
    """
    Get a nucleotide structure with a complex fold, to include a variety of aromatic
    ring interactions.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(Path(data_dir("structure")) / "4gxy.bcif")
    atoms = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    atoms = atoms[struc.filter_nucleotides(atoms)]

    # The CCD does not flag the 6-cycles in nucleobases as aromatic -> correct this
    aromatic_atom_names = [
        element + str(number)
        for element, number in itertools.product(["C", "N"], range(1, 10))
    ]
    bond_array = atoms.bonds.as_array()
    # Convert single and double bonds between those atoms into aromatic bonds
    aromatic_atom_mask = np.isin(
        atoms.atom_name[bond_array[:, 0]], aromatic_atom_names
    ) & np.isin(atoms.atom_name[bond_array[:, 1]], aromatic_atom_names)
    bond_array[aromatic_atom_mask, 2] = struc.BondType.AROMATIC
    atoms.bonds = struc.BondList(atoms.array_length(), bond_array)

    return atoms


@pytest.mark.parametrize(
    "res_name, ref_ring_members",
    [
        # No aromatic rings at all
        ("ALA", []),
        # Rings, but no aromatic ones
        ("GLC", []),
        # One aromatic ring
        (
            "TYR",
            [
                ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
            ],
        ),
        # Aromatic ring with heteroatoms
        (
            "HIS",
            [
                ("CG", "CD2", "NE2", "CE1", "ND1"),
            ],
        ),
        # Two fused aromatic rings
        (
            "TRP",
            [
                ("CG", "CD1", "CD2", "CE2", "NE1"),
                ("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"),
            ],
        ),
        # Disconnected aromatic rings
        (
            "BP5",
            [
                ("N1", "C1", "C2", "C3", "C4", "C5"),
                ("N2", "C6", "C7", "C8", "C9", "C11"),
            ],
        ),
    ],
    # Keep only the residue name as ID
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_find_known_aromatic_rings(res_name, ref_ring_members):
    """
    Check if aromatic rings are correctly identified by :func:`find_aromatic_rings()` in
    known molecules.
    """
    molecule = info.residue(res_name)
    rings = struc.find_aromatic_rings(molecule)
    test_ring_members = set(
        [frozenset(molecule.atom_name[atom_indices].tolist()) for atom_indices in rings]
    )
    ref_ring_members = set([frozenset(ring) for ring in ref_ring_members])

    assert test_ring_members == ref_ring_members


@pytest.mark.parametrize(
    "stacking_type, included_interactions, excluded_interactions",
    [
        (
            struc.PiStacking.PARALLEL,
            [
                (13, 14),
                (32, 33),
                (109, 121),
            ],
            [
                (109, 120),
                (109, 122),
            ],
        ),
        (
            struc.PiStacking.PERPENDICULAR,
            [
                (99, 100),
                (125, 126),
            ],
            [
                (16, 30),
                (97, 130),
                (16, 29),
            ],
        ),
    ],
    ids=lambda x: x.name if isinstance(x, IntEnum) else None,
)
def test_find_known_stacking_interactions(
    riboswitch_structure, stacking_type, included_interactions, excluded_interactions
):
    """
    Check if :func:`find_stacking_interactions()` correctly identifies pi-stacking
    interactions in a known complex folded nucleic acid structure.
    Due to the high number of interactions, check this exemplarily, i.e.
    check if interactions between certain residues are reported and others are
    definitely absent.
    """
    interactions = struc.find_stacking_interactions(
        riboswitch_structure,
    )
    interaction_res_ids = []
    for ring_indices_1, ring_indices_2, s_type in interactions:
        if s_type == stacking_type:
            interaction_res_ids.append(
                frozenset(
                    (
                        # Taking the first atom index is sufficient,
                        # as all atoms in the same ring are part of the same residue
                        riboswitch_structure.res_id[ring_indices_1[0]],
                        riboswitch_structure.res_id[ring_indices_2[0]],
                    )
                )
            )

    included_interactions = set(
        [frozenset(interaction) for interaction in included_interactions]
    )
    excluded_interactions = set(
        [frozenset(interaction) for interaction in excluded_interactions]
    )

    assert included_interactions.issubset(interaction_res_ids)
    assert excluded_interactions.isdisjoint(interaction_res_ids)


def test_no_duplicate_stacking_interactions(riboswitch_structure):
    """
    Check if :func:`find_stacking_interactions()` does not report duplicate
    interactions.
    """
    interactions = struc.find_stacking_interactions(riboswitch_structure)
    original_length = len(interactions)

    interactions = set(
        [
            frozenset((frozenset(ring_indices_1), frozenset(ring_indices_2)))
            for ring_indices_1, ring_indices_2, _ in interactions
        ]
    )
    deduplicated_length = len(interactions)

    assert deduplicated_length == original_length


def test_no_adjacent_stacking_interactions():
    """
    Ensure that :func:`find_stacking_interactions()` does not report interactions
    between adjacent (fused) aromatic rings.
    """
    # Tryptophan contains two fused aromatic rings
    molecule = info.residue("TRP")
    interactions = struc.find_stacking_interactions(molecule)

    assert len(interactions) == 0


def test_find_pi_cation_interactions():
    """
    Test π-cation interaction detection between aromatic residues and charged ligands.
    Uses PDB 3wip known to have pi-cation interactions between tryptophan/tyrosine
    residues and acetylcholine (ACH) ligand.
    """
    pdbx_file = pdbx.CIFFile.read(Path(data_dir("structure")) / "3wip.cif")
    atoms = pdbx.get_structure(
        pdbx_file, model=1, include_bonds=True, extra_fields=["charge"]
    )

    interactions = struc.find_pi_cation_interactions(atoms)

    assert len(interactions) > 0, "No pi-cation interactions found"

    # Assert interactions are between aromatic residues (TRP/TYR) and ACH ligand
    valid_aromatic_residues = {"TRP", "TYR"}
    for ring_atom_indices, cation_index in interactions:
        # Check ring atoms are from aromatic residues
        ring_residue = atoms.res_name[ring_atom_indices[0]]
        res_id = atoms.res_id[ring_atom_indices[0]]
        print(ring_residue)
        print(res_id)
        assert ring_residue in valid_aromatic_residues, (
            f"Found ring interaction with unexpected residue: {ring_residue}"
        )

        # Check cation is from ACH ligand
        cation_residue = atoms.res_name[cation_index]
        assert cation_residue == "ACH", (
            f"Found cation interaction with unexpected ligand: {cation_residue}"
        )

        # Verify the cation atom has positive charge
        cation_charge = atoms.charge[cation_index]
        assert cation_charge > 0, (
            f"Cation atom {atoms.atom_name[cation_index]} has non-positive charge: {cation_charge}"
        )
