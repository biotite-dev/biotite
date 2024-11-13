# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.fixture
def stack():
    stack = struc.AtomArrayStack(depth=3, length=5)
    stack.coord = np.arange(45).reshape((3, 5, 3))
    return stack


@pytest.mark.parametrize("as_coord", [False, True])
def test_rmsd(stack, as_coord):
    if as_coord:
        stack = stack.coord
    assert struc.rmsd(stack[0], stack).tolist() == pytest.approx(
        [0.0, 25.98076211, 51.96152423]
    )
    assert struc.rmsd(stack[0], stack[1]) == pytest.approx(25.9807621135)


@pytest.mark.parametrize("as_coord", [False, True])
def test_rmsf(stack, as_coord):
    if as_coord:
        stack = stack.coord
    assert struc.rmsf(struc.average(stack), stack).tolist() == pytest.approx(
        [21.21320344] * 5
    )


@pytest.fixture(scope="module")
def models():
    """
    Get a short multi-model structure
    """
    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), "1l2y.bcif"))
    return pdbx.get_structure(pdbx_file)


@pytest.fixture
def superimposed_models(models):
    """
    Models superimposed on the first model
    """
    bb_mask = struc.filter_peptide_backbone(models[0])
    supimp, _ = struc.superimpose(models[0], models, atom_mask=bb_mask)
    return models, supimp


def test_rmsd_gmx(superimposed_models):
    """
    Comparison of RMSD values computed with Biotite with results
    obtained from GROMACS 2021.5.
    """
    stack, supimp = superimposed_models
    rmsd = struc.rmsd(stack[0], supimp) / 10

    # Gromacs RMSDs -> Without mass-weighting:
    # echo "Backbone Protein" | \
    # gmx rms -s 1l2y.gro -f 1l2y.xtc -o rmsd.xvg -mw no
    rmsd_gmx = np.array(
        [
            0.0005037,
            0.1957698,
            0.2119313,
            0.2226127,
            0.184382,
            0.2210998,
            0.2712815,
            0.1372861,
            0.2348654,
            0.1848784,
            0.1893576,
            0.2500543,
            0.1946374,
            0.2101624,
            0.2180645,
            0.1836762,
            0.1681345,
            0.2363865,
            0.2287371,
            0.2546207,
            0.1604872,
            0.2167119,
            0.2176063,
            0.2069806,
            0.2535706,
            0.2682233,
            0.2252388,
            0.2419151,
            0.2343987,
            0.1902994,
            0.2334525,
            0.2010523,
            0.215444,
            0.1786632,
            0.2652018,
            0.174061,
            0.2591569,
            0.2602662,
        ]
    )

    assert np.allclose(rmsd, rmsd_gmx, atol=1e-03)


def test_rmspd_gmx(superimposed_models):
    """
    Comparison of the RMSPD computed with Biotite with results
    obtained from GROMACS 2021.5.
    """
    stack, _ = superimposed_models
    rmspd = struc.rmspd(stack[0], stack) / 10

    # Gromacs RMSDist:
    # echo "Protein" | \
    # gmx rmsdist -f 1l2y.xtc -s 1l2y.gro -o rmsdist.xvg -sumh no -pbc no
    rmspd_gmx = np.array(
        [
            0.000401147,
            0.125482,
            0.138913,
            0.138847,
            0.113917,
            0.132915,
            0.173084,
            0.103089,
            0.156309,
            0.114694,
            0.12964,
            0.15875,
            0.12876,
            0.128983,
            0.137031,
            0.126059,
            0.106726,
            0.154244,
            0.144405,
            0.174041,
            0.10417,
            0.130936,
            0.141216,
            0.125559,
            0.171342,
            0.165306,
            0.137616,
            0.154447,
            0.146337,
            0.116433,
            0.154976,
            0.128477,
            0.150537,
            0.111494,
            0.173234,
            0.116638,
            0.169524,
            0.15953,
        ]
    )

    assert np.allclose(rmspd, rmspd_gmx, atol=1e-03)


def test_rmsf_gmx(superimposed_models):
    """
    Comparison of RMSF values computed with Biotite with results
    obtained from GROMACS 2021.5.
    """
    stack, supimp = superimposed_models
    ca_mask = (stack[0].atom_name == "CA") & (stack[0].element == "C")
    rmsf = struc.rmsf(struc.average(supimp[:, ca_mask]), supimp[:, ca_mask]) / 10

    # Gromacs RMSF:
    # echo "C-alpha" | gmx rmsf -s 1l2y.gro -f 1l2y.xtc -o rmsf.xvg -res
    rmsf_gmx = np.array(
        [
            0.1379,
            0.036,
            0.0261,
            0.0255,
            0.029,
            0.0204,
            0.0199,
            0.0317,
            0.0365,
            0.0249,
            0.0269,
            0.032,
            0.0356,
            0.0446,
            0.059,
            0.037,
            0.0331,
            0.0392,
            0.0403,
            0.0954,
        ]
    )

    assert np.allclose(rmsf, rmsf_gmx, atol=1e-02)


@pytest.mark.parametrize(
    "multi_model, as_coord", itertools.product([False, True], [False, True])
)
def test_lddt_perfect(models, multi_model, as_coord):
    """
    Check if the lDDT of a structure with itself as reference is 1.0
    """
    reference = models[0]
    if multi_model:
        subject = models[0:1]
    else:
        subject = models[0]

    if as_coord:
        reference = reference.coord
        subject = subject.coord
        # Coordinates can only be used if 'exclude_same_residue' is False
        exclude_same_residue = False
    else:
        exclude_same_residue = True

    lddt = struc.lddt(reference, subject, exclude_same_residue=exclude_same_residue)

    if multi_model:
        assert lddt.tolist() == [1.0]
    else:
        assert lddt == 1.0


def test_lddt_consistency(models):
    """
    Check if the computed lDDT via :func:`lddt` is the same as the reference value from
    the https://swissmodel.expasy.org/assess web server.
    """
    # As each model needs to be uploaded separately, only the first 8 models were run
    REFERENCE_LDDT = [
        1.00,
        0.86,
        0.83,
        0.85,
        0.88,
        0.85,
        0.81,
        0.89,
    ]

    # The web server computes the lDDT without hydrogen atoms
    models = models[:, models.element != "H"]

    reference = models[0]
    subject = models[: len(REFERENCE_LDDT)]
    lddt = struc.lddt(reference, subject)

    assert lddt.tolist() == pytest.approx(REFERENCE_LDDT, abs=1e-2)


@pytest.mark.parametrize("multi_model", [False, True])
def test_lddt_custom_aggregation(models, multi_model):
    """
    Check if custom `aggregation` in :func:`lddt` works by giving a custom aggregation
    that simply aggregates all values, i.e. it should give the same result as the
    `all` aggregation.
    """
    reference = models[0]
    if multi_model:
        subject = models
    else:
        subject = models[1]
    # Every atom goes into the same aggregation bin
    aggregation_bins = np.zeros(models.array_length(), dtype=int)

    ref_lddt = struc.lddt(reference, subject, aggregation="all")

    test_lddt = struc.lddt(reference, subject, aggregation=aggregation_bins)

    # There is only one aggregation bin -> only one lDDT value (for each value)
    assert test_lddt.shape[-1] == 1
    assert test_lddt[..., 0].tolist() == ref_lddt.tolist()


@pytest.mark.parametrize(
    "multi_model, aggregation",
    itertools.product([False, True], ["chain", "residue", "atom"]),
)
def test_lddt_aggregation_levels(models, multi_model, aggregation):
    """
    Check if the predefined aggregation levels :func:`lddt()` return the expected shape.
    Furthermore, the average of each aggregated lDDT values should be similar to the
    global lDDT value (not exactly, as the global lDDT is weighted by the number of
    contacts in each residue/chain).
    """
    ABS_ERROR = 0.02

    reference = models[0]
    if multi_model:
        subject = models
    else:
        subject = models[1]

    expected_shape = ()
    if multi_model:
        expected_shape = expected_shape + (subject.stack_depth(),)
    if aggregation == "chain":
        expected_shape = expected_shape + (struc.get_chain_count(subject),)
    elif aggregation == "residue":
        expected_shape = expected_shape + (struc.get_residue_count(subject),)
    elif aggregation == "atom":
        expected_shape = expected_shape + (models.array_length(),)

    lddt = struc.lddt(reference, subject, aggregation=aggregation)
    all_lddt = struc.lddt(reference, subject, aggregation="all")

    assert lddt.shape == expected_shape
    if multi_model:
        assert np.mean(lddt, axis=-1).tolist() == pytest.approx(
            all_lddt.tolist(), abs=ABS_ERROR
        )
    else:
        assert np.mean(lddt) == pytest.approx(all_lddt, abs=ABS_ERROR)


# Dividing by zero contacts for masked atoms would normally raise a warning
@pytest.mark.filterwarnings("error:.*invalid value encountered in divide")
@pytest.mark.parametrize("seed", range(10))
def test_lddt_mask(models, seed):
    """
    Using the same `subject_mask` and `partner_mask` should give the same score as
    using the same mask on the structure directly, before it is passed to
    :func:`lddt()`.
    """
    rng = np.random.default_rng(seed)
    mask = rng.choice([False, True], size=models.array_length())

    reference = models[0]
    subject = models[1]

    # Do not aggregate to make the test more strict
    ref_lddt = struc.lddt(reference[mask], subject[mask], aggregation="atom")

    test_lddt = struc.lddt(
        reference, subject, aggregation="atom", atom_mask=mask, partner_mask=mask
    )

    # The subject mask defines for which atoms the contact partners should be identified
    # In consequence, for all atoms that are masked out, the lDDT should be NaN
    assert np.isnan(test_lddt[..., ~mask]).all()
    test_lddt = test_lddt[mask]
    assert test_lddt.tolist() == pytest.approx(ref_lddt.tolist())


def test_custom_lddt_symmetric(models):
    """
    Check that in :func:`lddt()` with ``symmetric=True`` the *lDDT* score is independent
    of the order of the structures.
    """
    model1 = models[0]
    model2 = models[1]

    # Do not aggregate to make the test more strict
    model1_symmetric_lddt = struc.lddt(model1, model2, "atom", symmetric=True)
    model2_symmetric_lddt = struc.lddt(model2, model1, "atom", symmetric=True)

    # The order of the structures should not matter
    assert model1_symmetric_lddt.tolist() == pytest.approx(
        model2_symmetric_lddt.tolist()
    )


@pytest.mark.parametrize(
    "what_is_nan, ref_lddt",
    [
        # No reference coordinates (except one atom)
        # -> set of reference contacts is empty
        ("reference", np.nan),
        # No subject coordinates (except one atom)
        # -> no recovered contacts
        ("subject", 0.0),
    ],
)
def test_lddt_with_missing_atoms(models, what_is_nan, ref_lddt):
    """
    Check if NaN values in the coordinates are handled correctly.
    """
    reference = models[0].copy()
    subject = models[1].copy()
    if what_is_nan == "reference":
        reference.coord[1:, :] = np.nan
    elif what_is_nan == "subject":
        subject.coord[1:, :] = np.nan

    test_lddt = struc.lddt(reference, subject)

    # Use 'np.array_equal()' to handle equality of NaN values
    assert np.array_equal(test_lddt, ref_lddt, equal_nan=True)
