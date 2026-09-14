# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir

# Elements supported by `infer_bond_types()`
_ORGANIC_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "SI", "P", "S", "CL", "SE", "BR", "I",
}  # fmt: skip
# Molecules with sulfur or selenium in each possible number of bond partners,
# as the allowed valences of these elements are handled specially
_SULFUR_COMPOUNDS = [
    "IS8",  # thiocarbonyl (1 partner, valence 2)
    "OPS",  # thiolate (1 partner, valence 1)
    "CYS",  # thiol (2 partners, valence 2)
    "MET",  # thioether (2 partners, valence 2)
    "ZVH",  # thiophene (2 partners, valence 2)
    "MSE",  # selenoether (2 partners, valence 2)
    "NK2",  # sulfonium (3 partners, valence 3)
    "DMS",  # sulfoxide (3 partners, valence 4)
    "BLT",  # selenonium (3 partners, valence 3)
    "L2L",  # sulfone (4 partners, valence 6)
]
# Molecules that require valence states with a formal charge
_CHARGED_COMPOUNDS = [
    "CMO",  # carbon monoxide
    "MNC",  # isocyanide
    "2QB",  # azide
    "JYE",  # tetrazole
    "DL6",  # nitro group
    "7RD",  # charged nitrogen
    "A9J",  # hexafluorophosphate
]


def _sample_ccd_residues(size):
    """
    Randomly sample names of released components from the CCD.
    """
    chem_comp = info.get_ccd()["chem_comp"]
    residue_names = chem_comp["id"].as_array()
    release_status = chem_comp["pdbx_release_status"].as_array()
    residue_names = residue_names[release_status == "REL"]
    return np.random.default_rng(0).choice(residue_names, size=size).tolist()


def _without_overlong_bonds(atoms, max_length=3.0):
    """
    Get the bonds of the given atoms, excluding bonds that are longer than the given
    maximum length, as these indicate defective coordinates.
    """
    bond_array = atoms.bonds.as_array()
    bond_lengths = struc.distance(
        atoms.coord[bond_array[:, 0]], atoms.coord[bond_array[:, 1]]
    )
    return struc.BondList(atoms.array_length(), bond_array[bond_lengths <= max_length])


def _without_metal_bonds(atoms):
    """
    Get the bonds of the given atoms, excluding bonds involving a metal atom.
    """
    is_metal = ~np.isin(atoms.element, list(_ORGANIC_ELEMENTS))
    bond_array = atoms.bonds.as_array()
    is_metal_bond = is_metal[bond_array[:, 0]] | is_metal[bond_array[:, 1]]
    return struc.BondList(atoms.array_length(), bond_array[~is_metal_bond])


def _infer_aromaticity(atoms, bonds):
    """
    Set the type of each bond that *RDKit* perceives as aromatic to
    :attr:`BondType.AROMATIC`.

    Applied to two bond lists of the same molecule, this removes the ambiguity, which
    of the equivalent resonance structures was chosen for kekulized aromatic systems.
    """
    # Import locally in case the `rdkit`package is absent
    from rdkit import Chem
    import biotite.interface.rdkit as rdkit_interface

    molecule = atoms.copy()
    molecule.bonds = bonds
    mol = rdkit_interface.to_mol(molecule)
    # Perceive only rings and aromaticity, as the valence checks of a full sanitization
    # may fail for the resonance structure at hand
    Chem.SanitizeMol(
        mol, Chem.SANITIZE_SYMMRINGS | Chem.SANITIZE_SETAROMATICITY, catchErrors=True
    )

    bond_array = bonds.as_array()
    aromatic_bond_array = bond_array.copy()
    for k, (atom_i, atom_j, _) in enumerate(bond_array):
        if mol.GetBondBetweenAtoms(int(atom_i), int(atom_j)).GetIsAromatic():
            aromatic_bond_array[k, 2] = struc.BondType.AROMATIC
    return struc.BondList(atoms.array_length(), aromatic_bond_array)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("as_mask", [False, True])
def test_find_connected(seed, as_mask):
    """
    The ``label_asym_id`` in PDBx files distinguishes different molecules.
    Therefore, all connected atoms should hame the same chain ID.
    An exception are water molecules, which have the same chain ID, albeit not being
    bonded to each other.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(data_dir("structure") / "pdb" / "1k6p.bcif")
    atoms = pdbx.get_structure(
        pdbx_file, model=1, include_bonds=True, use_author_fields=False
    )
    atoms = atoms[~struc.filter_solvent(atoms)]

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
    ["res_name", "expected_bonds"],
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
    ["cif_path", "expected_bond_indices"],
    [
        (
            data_dir("structure") / "pdb" / "3o5r.cif",
            [252, 257],  # Carbonyl carbon and subsequent backbone nitrogen
        )
    ],
    ids=["3o5r.cif"],
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


@pytest.mark.filterwarnings("ignore:.*coordinates are missing.*")
@pytest.mark.parametrize("res_name", _sample_ccd_residues(100))
def test_connect_via_distances_small_molecules(res_name):
    """
    Expect that the connectivity of small molecules from the CCD is recovered by
    :func:`connect_via_distances()` from their idealized coordinates.
    """
    ref_atoms = info.residue(res_name, allow_missing_coord=True)
    if np.any(ref_atoms.element == "X"):
        pytest.skip("Molecule contains an atom with unknown element")
    if np.isnan(ref_atoms.coord).any():
        pytest.skip("Molecule has missing coordinates")
    # Some CCD entries have defective ideal coordinates, i.e. bonded atoms are far apart
    # -> Such bonds cannot be detected from distances and are removed from the reference
    ref_atoms.bonds = _without_overlong_bonds(ref_atoms)
    # Only the connectivity can be detected from distances, not the bond order
    ref_atoms.bonds.remove_bond_order()

    test_atoms = ref_atoms.copy()
    test_atoms.bonds = struc.connect_via_distances(test_atoms)

    # Coordination bonds cannot be told apart from mere proximity by covalent radii:
    # In metal clusters (e.g. iron-sulfur clusters) the metal atoms are within
    # covalent distance of each other, although the CCD defines no bonds between them,
    # and conversely a ligand atom may be within covalent distance of a metal without
    # being coordinated to it
    # -> Bonds involving metal atoms are not compared
    ref_bonds = _without_metal_bonds(ref_atoms)
    test_bonds = _without_metal_bonds(test_atoms)
    assert test_bonds.as_set() == ref_bonds.as_set()


@pytest.mark.filterwarnings("ignore:.*coordinates are missing.*")
@pytest.mark.filterwarnings("error::biotite.structure.InconsistentBondTypeWarning")
@pytest.mark.parametrize("use_charge", [False, True])
@pytest.mark.parametrize(
    "res_name",
    # Supplement the random sample with molecules that need special valence states
    sorted(
        set(_sample_ccd_residues(100))
        | set(_SULFUR_COMPOUNDS)
        | set(_CHARGED_COMPOUNDS)
    ),
)
def test_infer_bond_types_small_molecules(res_name, use_charge):
    """
    Expect that the bond types of small molecules from the CCD are recovered by
    :func:`infer_bond_types()` from their connectivity.

    As the algorithm may choose a different, but equivalent resonance structure than
    the reference, the bonds of aromatic systems are set to :attr:`BondType.AROMATIC`
    on both sides, before they are compared.

    The known formal charges merely constrain the search, so the same bond types are
    expected, regardless of whether they are given.
    """
    pytest.importorskip("rdkit")

    ref_atoms = info.residue(res_name, allow_missing_coord=True)
    if not set(ref_atoms.element) <= _ORGANIC_ELEMENTS:
        pytest.skip("Molecule contains unsupported elements")
    # The reference uses aromatic bond types
    # -> kekulize it, to give it the same treatment as the inferred bonds,
    # as the CCD and rdkit may disagree on aromaticity definition
    ref_bonds = ref_atoms.bonds.copy()
    ref_bonds.remove_aromaticity()
    # Bonds that cannot be kekulized are converted to `BondType.ANY`
    if (ref_bonds.as_array()[:, 2] == struc.BondType.ANY).any():
        pytest.skip("Reference molecule has undefined bond types")
    atoms = ref_atoms.copy()
    connectivity = ref_atoms.bonds.copy()
    connectivity.remove_bond_order()
    atoms.bonds = connectivity
    if not use_charge:
        atoms.del_annotation("charge")

    # Bound the number of iterations to fail instead of stalling,
    # if no assignment is found
    test_bonds, _ = struc.infer_bond_types(
        atoms, total_charge=int(ref_atoms.charge.sum()), max_iterations=1_000_000
    )

    test_bonds = _infer_aromaticity(ref_atoms, test_bonds)
    ref_bonds = _infer_aromaticity(ref_atoms, ref_bonds)
    # Equivalent terminal atoms, e.g. the oxygen atoms of a carboxylate group, can be
    # assigned to different bond orders, which is a valid alternative resonance
    # structure
    differing_bonds = ref_bonds.as_array()[
        test_bonds.as_array()[:, 2] != ref_bonds.as_array()[:, 2]
    ][:, :2]
    n_partners = np.count_nonzero(ref_atoms.bonds.get_all_bonds()[0] != -1, axis=1)
    if len(differing_bonds) > 0 and np.all(
        np.any(n_partners[differing_bonds] == 1, axis=1)
    ):
        pytest.skip("Bond orders of equivalent terminal atoms are ambiguous")

    assert test_bonds.as_set() == ref_bonds.as_set()


@pytest.mark.parametrize("periodic", [False, True])
def test_method_consistency(periodic):
    """
    Check if :func:`connect_via_distances()` and
    :func:`connect_via_residue_names()` give the same bond list
    """
    # Structure with peptide, nucleotide, small molecules and water
    pdbx_file = pdbx.BinaryCIFFile.read(data_dir("structure") / "pdb" / "5ugo.bcif")
    atoms = pdbx.get_structure(pdbx_file, model=1)
    if periodic:
        # Add large dummy box to test parameter
        # No actual bonds over the periodic boundary are expected
        atoms.box = np.identity(3) * 100

    bonds_from_names = struc.connect_via_residue_names(atoms)
    bonds_from_names.remove_bond_order()

    bonds_from_distances = struc.connect_via_distances(atoms, periodic=periodic)

    assert bonds_from_distances.as_set() == bonds_from_names.as_set()
