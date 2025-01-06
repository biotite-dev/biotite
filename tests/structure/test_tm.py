import json
from pathlib import Path
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.fixture
def without_tm_gap_penalty():
    """
    Set the gap penalty for the iterative alignment step to 0
    """
    import biotite.structure.tm

    original_gap_penalty = biotite.structure.tm._TM_GAP_PENALTY
    biotite.structure.tm._TM_GAP_PENALTY = 0
    yield
    biotite.structure.tm._TM_GAP_PENALTY = original_gap_penalty


@pytest.mark.parametrize("reference_length", ["shorter", "longer", "reference", 20])
def test_tm_score_perfect(reference_length):
    """
    Check if the TM-score of a structure with itself as reference is 1.0.

    Test different reference lengths here as well.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(Path(data_dir("structure")) / "1l2y.bcif")
    atoms = pdbx.get_structure(pdbx_file, model=1)
    ca_indices = np.where(atoms.atom_name == "CA")[0]

    assert struc.tm_score(atoms, atoms, ca_indices, ca_indices, reference_length) == 1.0


@pytest.mark.parametrize("pdb_id", ["1l2y", "1gya"])
def test_tm_score_consistency(pdb_id):
    """
    Check if the TM-score is correctly computed by comparing it to the result of
    *USalign*.
    To decouple TM-score calculation from :func:`superimpose_structural_homologs()`,
    the TM-score is calculated for two models of the same length.
    """
    with open(Path(data_dir("structure")) / "tm" / "tm_scores.json", "r") as file:
        ref_tm_scores = json.load(file)[pdb_id]

    pdbx_file = pdbx.BinaryCIFFile.read(Path(data_dir("structure")) / f"{pdb_id}.bcif")
    atoms = pdbx.get_structure(pdbx_file)
    atoms = atoms[:, struc.filter_amino_acids(atoms)]
    reference = atoms[0]
    ca_indices = np.where(atoms.atom_name == "CA")[0]

    test_tm_scores = [
        struc.tm_score(reference, atoms[i], ca_indices, ca_indices)
        for i in range(0, atoms.stack_depth())
    ]

    assert test_tm_scores == pytest.approx(ref_tm_scores, abs=1e-2)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("structural_alphabet", ["3Di", "PB"])
def test_superimpose_identical(without_tm_gap_penalty, seed, structural_alphabet):
    """
    Check if :func:`superimpose_structural_homologs()` is able to superimpose
    two identical complexes with randomized deletions.
    As the alignment should detect the deletions, the superimposed
    RMSD should be 0 and the TM-score should be 1.

    For the iterative alignment step the gap penalty is set to 0, to avoid the situation
    where non-corresponding residues are aligned to avoid the gap penalty.
    """
    P_CONSERVATION = 0.8

    pdbx_file = pdbx.BinaryCIFFile.read(Path(data_dir("structure")) / "1aki.bcif")
    atoms = pdbx.get_structure(pdbx_file, model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]

    # Delete random residues
    fixed = atoms
    rng = np.random.default_rng(seed)
    # Only delete residues in one structure to avoid cases where non-corresponding
    # residues are rightfully aligned, for example the deletions
    #     01234-6789
    #     0123-56789
    # would align to
    #     012346789
    #     012356789
    # The non-corresponding positions '4' and '5' would align to each other in this case
    mobile = _delete_random_residues(atoms, rng, P_CONSERVATION)
    # Randomly move structure to increase the challenge
    mobile.coord = _transform_random_affine(mobile.coord, rng)

    super, _, fix_indices, mob_indices = struc.superimpose_structural_homologs(
        # Define max-iterations to avoid infinite loop if something goes wrong
        fixed,
        mobile,
        structural_alphabet,
        max_iterations=100,
    )

    # The superimposition anchors must be CA atoms
    assert np.all(fixed.atom_name[fix_indices] == "CA")
    assert np.all(mobile.atom_name[mob_indices] == "CA")
    # Expect that the found corresponding residues are actually the same residues from
    # the original structure in most cases
    assert fixed.res_id[fix_indices].tolist() == mobile.res_id[mob_indices].tolist()
    assert struc.tm_score(fixed, super, fix_indices, mob_indices) == pytest.approx(
        1.0, abs=1e-3
    )
    assert struc.rmsd(fixed[fix_indices], super[mob_indices]) == pytest.approx(
        0.0, abs=1e-3
    )


@pytest.mark.parametrize(
    "fixed_pdb_id, mobile_pdb_id, ref_tm_score",
    [
        ("1p4k", "4osx", 0.87),
        ("3lsj", "3rd3", 0.78),
        ("2nwd", "1hml", 0.91),
        ("3kcs", "6oa8", 0.93),
        ("2nwd", "1qgi", 0.55),
        ("3kcs", "1gl4", 0.82),
    ],
)
def test_superimpose_consistency(fixed_pdb_id, mobile_pdb_id, ref_tm_score):
    """
    Check if two complexes with high structure similarity, can be properly superimposed
    with :func:`superimpose_structural_homologs()`, even if sequence homology is low.
    The performance is evaluated in terms of the TM-score compared to the result of
    *US-align*.

    The chosen structure pairs have at least a TM-score of 0.5 according to *US-align*.
    This ensures that the structures have 'about the same fold' and therefore the
    superimposition is not spurious.

    *US-align* is used instead of *TM-align* to be able to align multimeric structures.
    """
    # Sometimes US-align might perform slightly better
    SCORE_TOLERANCE = 0.05

    fixed = _get_peptide_assembly(
        Path(data_dir("structure")) / "homologs" / f"{fixed_pdb_id}.bcif"
    )
    mobile = _get_peptide_assembly(
        Path(data_dir("structure")) / "homologs" / f"{mobile_pdb_id}.bcif"
    )

    super, _, fix_indices, mob_indices = struc.superimpose_structural_homologs(
        fixed,
        mobile,
    )
    assert (
        struc.tm_score(fixed, super, fix_indices, mob_indices)
        >= ref_tm_score - SCORE_TOLERANCE
    )


def _transform_random_affine(coord, rng):
    coord = struc.translate(coord, rng.uniform(low=0, high=10, size=3))
    coord = struc.rotate(coord, rng.uniform(low=0, high=2 * np.pi, size=3))
    return coord


def _delete_random_residues(atoms, rng, p_conservation):
    residue_starts = struc.get_residue_starts(atoms)
    conserved_residue_starts = rng.choice(
        residue_starts, size=int(p_conservation * len(residue_starts)), replace=False
    )
    conservation_mask = np.any(
        struc.get_residue_masks(atoms, conserved_residue_starts), axis=0
    )
    return atoms[..., conservation_mask]


def _get_peptide_assembly(bcif_file_path):
    """
    Load assembly from a BinaryCIF file and filter peptide residues.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(bcif_file_path)
    atoms = pdbx.get_assembly(pdbx_file, model=1)
    return atoms[struc.filter_amino_acids(atoms)]
