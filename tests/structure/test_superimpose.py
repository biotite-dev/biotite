# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
import itertools
from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
from tests.util import data_dir


@pytest.mark.parametrize(
    "seed, multi_model", itertools.product(range(10), [False, True])
)
def test_restoration(seed, multi_model):
    """
    Check if randomly relocated coordinates can be restored to their
    original position by superimposition.
    """
    N_MODELS = 10
    N_COORD = 100

    np.random.seed(seed)
    if multi_model:
        ref_coord = np.random.rand(N_MODELS, N_COORD, 3)
    else:
        ref_coord = np.random.rand(N_COORD, 3)
    ref_coord = _transform_random_affine(ref_coord)

    # Try to restore original coordinates after random relocation
    test_coord = _transform_random_affine(ref_coord)
    test_coord, _ = struc.superimpose(ref_coord, test_coord)

    assert test_coord.flatten().tolist() == pytest.approx(
        ref_coord.flatten().tolist(), abs=1e-6
    )


def test_rotation_matrix():
    """
    Create randomly generated coordinates *a* and rotate them via a
    rotation matrix to obtain new coordinates *b*.
    ``superimpose(b, a)`` should give the rotation matrix.
    """
    N_COORD = 100

    # A rotation matrix that rotates 90 degrees around the z-axis
    ref_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    np.random.seed(0)
    original_coord = np.random.rand(N_COORD, 3)
    # Rotate about 90 degrees around z-axis
    rotated_coord = struc.rotate(original_coord, angles=(0, 0, np.pi / 2))
    _, transform = struc.superimpose(rotated_coord, original_coord)
    test_rotation = transform.rotation

    assert test_rotation.flatten().tolist() == pytest.approx(
        ref_rotation.flatten().tolist(), abs=1e-6
    )


@pytest.mark.parametrize(
    "path, coord_only",
    itertools.product(glob.glob(join(data_dir("structure"), "*.bcif")), [False, True]),
)
def test_superimposition_array(path, coord_only):
    """
    Take a structure and rotate and translate a copy of it, so that they
    are not superimposed anymore.
    Then superimpose these structure onto each other and expect an
    almost perfect match.
    """
    fixed = strucio.load_structure(path, model=1)

    mobile = fixed.copy()
    mobile = struc.rotate(mobile, (1, 2, 3))
    mobile = struc.translate(mobile, (1, 2, 3))

    if coord_only:
        fixed = fixed.coord
        mobile = mobile.coord

    fitted, transformation = struc.superimpose(fixed, mobile)

    if coord_only:
        assert isinstance(fitted, np.ndarray)
    assert struc.rmsd(fixed, fitted) == pytest.approx(0, abs=6e-4)

    fitted = transformation.apply(mobile)

    if coord_only:
        assert isinstance(fitted, np.ndarray)
    assert struc.rmsd(fixed, fitted) == pytest.approx(0, abs=6e-4)


@pytest.mark.parametrize("ca_only", (True, False))
def test_superimposition_stack(ca_only):
    """
    Take a structure with multiple models where each model is not
    (optimally) superimposed onto each other.
    Then superimpose and expect an improved RMSD.
    """
    path = join(data_dir("structure"), "1l2y.bcif")
    stack = strucio.load_structure(path)
    fixed = stack[0]
    mobile = stack[1:]
    if ca_only:
        mask = mobile.atom_name == "CA"
    else:
        mask = None

    fitted, _ = struc.superimpose(fixed, mobile, mask)

    if ca_only:
        # The superimpositions are better for most cases than the
        # superimpositions in the structure file
        # -> Use average
        assert np.mean(struc.rmsd(fixed, fitted)) < np.mean(struc.rmsd(fixed, mobile))
    else:
        # The superimpositions are better than the superimpositions
        # in the structure file
        assert (struc.rmsd(fixed, fitted) < struc.rmsd(fixed, mobile)).all()


@pytest.mark.parametrize("seed", range(5))
def test_masked_superimposition(seed):
    """
    Take two models of the same structure and superimpose based on a
    single, randomly chosen atom.
    Since two atoms can be superimposed perfectly, the distance between
    the atom in both models should be 0.
    """
    path = join(data_dir("structure"), "1l2y.bcif")
    fixed = strucio.load_structure(path, model=1)
    mobile = strucio.load_structure(path, model=2)

    # Create random mask for a single atom
    np.random.seed(seed)
    mask = np.full(fixed.array_length(), False)
    mask[np.random.randint(fixed.array_length())] = True

    # The distance between the atom in both models should not be
    # already 0 prior to superimposition
    assert struc.distance(fixed[mask], mobile[mask])[0] != pytest.approx(0, abs=5e-4)

    fitted, transformation = struc.superimpose(fixed, mobile, mask)

    assert struc.distance(fixed[mask], fitted[mask])[0] == pytest.approx(0, abs=5e-4)

    fitted = transformation.apply(mobile)

    struc.distance(fixed[mask], fitted[mask])[0] == pytest.approx(0, abs=5e-4)


@pytest.mark.parametrize(
    "single_model, single_atom", itertools.product([False, True], [False, True])
)
def test_input_shapes(single_model, single_atom):
    """
    Test whether :func:`superimpose()` infers the correct output shape,
    even if the input :class:`AtomArrayStack` contains only a single
    model or a single atom.
    """
    path = join(data_dir("structure"), "1l2y.bcif")
    stack = strucio.load_structure(path)
    fixed = stack[0]

    mobile = stack
    if single_model:
        mobile = mobile[:1, :]
    if single_atom:
        mobile = mobile[:, :1]
        fixed = fixed[:1]

    fitted, _ = struc.superimpose(fixed, mobile)

    assert isinstance(fitted, type(mobile))
    assert fitted.coord.shape == mobile.coord.shape


@pytest.mark.parametrize("seed", range(5))
def test_outlier_detection(seed):
    """
    Check if :meth:`superimpose_without_outliers()` is able to detect
    clear outliers in artificially generated coordinates.
    """
    N_COORD = 100
    # Keep P_OUTLIERS < 0.5, otherwise outliers would be superimposed
    P_OUTLIERS = 0.1
    # Keep NOISE << OUTLIER_MAGNITUDE
    NOISE = 0.1
    OUTLIER_MAGNITUDE = 100

    np.random.seed(seed)
    fixed_coord = np.random.rand(N_COORD, 3)
    # Add a little bit of noise
    mobile_coord = fixed_coord + np.random.rand(N_COORD, 3) * NOISE
    # Make some coordinates outliers
    ref_outlier_mask = np.random.choice(
        [False, True], size=N_COORD, p=[1 - P_OUTLIERS, P_OUTLIERS]
    )
    mobile_coord[ref_outlier_mask] += OUTLIER_MAGNITUDE
    mobile_coord = _transform_random_affine(mobile_coord)

    superimposed_coord, _, anchors = struc.superimpose_without_outliers(
        # Increase the threshold a bit,
        # to ensure that no inlier is classified as outlier
        fixed_coord,
        mobile_coord,
        outlier_threshold=3.0,
    )
    test_outlier_mask = np.full(N_COORD, True)
    test_outlier_mask[anchors] = False

    assert test_outlier_mask.tolist() == ref_outlier_mask.tolist()
    # Without the outliers, the RMSD should be in the noise range
    assert (
        struc.rmsd(
            fixed_coord[~ref_outlier_mask], superimposed_coord[~ref_outlier_mask]
        )
        < NOISE
    )


@pytest.mark.parametrize(
    "multi_model, coord_only", itertools.product([False, True], [False, True])
)
def test_superimpose_without_outliers_inputs(multi_model, coord_only):
    """
    Ensure that :meth:`superimpose_without_outliers()` is able to handle
    single models, multiple models and pure coordinates.
    """
    path = join(data_dir("structure"), "1l2y.bcif")
    atoms = strucio.load_structure(path)
    if not multi_model:
        atoms = atoms[0]
    if coord_only:
        atoms = atoms.coord

    superimposed, transform, _ = struc.superimpose_without_outliers(atoms, atoms)

    assert isinstance(superimposed, type(atoms))
    assert superimposed.shape == atoms.shape
    transform_matrix = transform.as_matrix()
    if multi_model:
        assert transform_matrix.shape == (atoms.shape[0], 4, 4)
    else:
        assert transform_matrix.shape == (1, 4, 4)
    # Fixed and mobile are the same
    # -> expect identity transformation and all atoms as anchors
    for matrix in transform_matrix:
        assert np.allclose(matrix, np.eye(4), atol=1e-6)


@pytest.mark.parametrize(
    "pdb_id, chain_id, as_stack",
    [
        ("1aki", "A", False),  # is a protein
        ("1aki", "A", True),
        ("4gxy", "A", False),  # is a nucleic acid
        ("4gxy", "A", True),
    ],
)
def test_superimpose_homologs(pdb_id, chain_id, as_stack):
    """
    Check if :meth:`superimpose_homologs()` is able to superimpose
    two homologous structures.
    The homologous structures originate from the same structure with
    randomized deletions.
    As the alignment should detect the deletions, the superimposed
    RMSD should be 0.
    """
    P_CONSERVATION = 0.9
    path = join(data_dir("structure"), f"{pdb_id}.bcif")
    atoms = strucio.load_structure(path)
    atoms = atoms[atoms.chain_id == chain_id]
    if as_stack:
        atoms = struc.stack([atoms])

    # Delete random residues
    np.random.seed(0)
    fixed_atoms = _delete_random_residues(atoms, P_CONSERVATION)
    mobile_atoms = _delete_random_residues(atoms, P_CONSERVATION)
    mobile_atoms.coord = _transform_random_affine(mobile_atoms.coord)

    super_atoms, _, fix_anchors, mob_anchors = struc.superimpose_homologs(
        fixed_atoms, mobile_atoms
    )

    # Check if corresponding residues were superimposed
    assert (
        fixed_atoms.res_id[fix_anchors].tolist()
        == mobile_atoms.res_id[mob_anchors].tolist()
    )
    # If a stack, it only contains one model
    if as_stack:
        fixed_atoms = fixed_atoms[0]
        super_atoms = super_atoms[0]
    assert struc.rmsd(
        fixed_atoms[..., fix_anchors], super_atoms[..., mob_anchors]
    ) == pytest.approx(0, abs=1e-3)


def _transform_random_affine(coord):
    coord = struc.translate(coord, np.random.rand(3))
    coord = struc.rotate(coord, np.random.uniform(low=0, high=2 * np.pi, size=3))
    return coord


def _delete_random_residues(atoms, p_conservation):
    residue_starts = struc.get_residue_starts(atoms)
    conserved_residue_starts = np.random.choice(
        residue_starts, size=int(p_conservation * len(residue_starts)), replace=False
    )
    conservation_mask = np.any(
        struc.get_residue_masks(atoms, conserved_residue_starts), axis=0
    )
    return atoms[..., conservation_mask]
