# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.info as strucinfo
import biotite.structure.io.pdbx as pdbx
from biotite.structure.info.ccd import _CCD_FILE as INTERNAL_CCD_FILE
from biotite.structure.io import load_structure
from tests.util import data_dir


@pytest.fixture
def fake_ccd_path(tmp_path):
    block = pdbx.BinaryCIFBlock()
    block["chem_comp"] = pdbx.BinaryCIFCategory({"id": "FOO", "name": "Foo"})
    block["chem_comp_atom"] = pdbx.BinaryCIFCategory({"comp_id": "FOO"})
    block["chem_comp_bond"] = pdbx.BinaryCIFCategory({"comp_id": "FOO"})
    file = pdbx.BinaryCIFFile()
    file["components"] = block
    original_path = INTERNAL_CCD_FILE
    path = tmp_path / "components.bcif"
    file.write(path)

    yield path

    # Restore the original internal CCD path
    strucinfo.set_ccd_path(original_path)


@pytest.mark.parametrize(
    "function, included, excluded",
    [
        (strucinfo.amino_acid_names, ["ALA", "ARG", "ASN", "ASP"], ["HOH"]),
        (strucinfo.nucleotide_names, ["A", "C", "G", "U"], ["HOH", "ALA"]),
        (strucinfo.carbohydrate_names, ["GLC", "RIB"], ["HOH", "ALA"]),
    ],
)
def test_group_names(function, included, excluded):
    """
    Check if some expected group names are present in the group list
    and unexpected ones are not.
    """
    # Run two times to check if caching works
    for _ in range(2):
        res_names = function()
        for res_name in included:
            assert res_name in res_names
        for res_name in excluded:
            assert res_name not in res_names


def test_mass():
    """
    Test whether the mass of a residue is the same as the sum of the
    masses of its contained atoms.
    """
    array = load_structure(join(data_dir("structure"), "1l2y.bcif"))[0]
    _, res_names = struc.get_residues(array)
    water_mass = strucinfo.mass("H") * 2 + strucinfo.mass("O")
    # Mass of water must be subtracted
    masses = [strucinfo.mass(res_name) - water_mass for res_name in res_names]
    # C-terminus normally has additional oxygen atom
    masses[-1] += strucinfo.mass("O")
    ref_masses = [strucinfo.mass(res) for res in struc.residue_iter(array)]
    # Up to three additional/missing hydrogens are allowed
    # (protonation state)
    mass_diff = np.abs(
        np.array([mass - ref_mass for mass, ref_mass in zip(masses, ref_masses)])
    )
    assert (mass_diff // strucinfo.mass("H") <= 3).all()
    # Check if the mass difference is a multiple of the hydrogen mass
    multiple_of_h_masses = mass_diff / strucinfo.mass("H")
    assert np.all(np.round(multiple_of_h_masses, decimals=2) % 1 == 0)


def test_protor_radii():
    """
    Assert that ProtOr VdW radii (except hydrogen) can be calculated for
    all atoms in the given structure, since the structure (1GYA)
    does only contain standard amino acids after the removal of
    glycosylation.
    This means, that none of the resulting radii should be the None.
    """
    array = load_structure(join(data_dir("structure"), "1gya.bcif"))
    array = array[..., array.element != "H"]
    array = array[..., struc.filter_amino_acids(array)]
    for res_name, atom_name in zip(array.res_name, array.atom_name):
        radius = strucinfo.vdw_radius_protor(res_name, atom_name)
        assert isinstance(radius, float)
        assert radius is not None


def test_protor_radii_invalid():
    with pytest.raises(ValueError):
        # Expect raised exception for hydrogen atoms
        strucinfo.vdw_radius_protor("FOO", "H1")
    with pytest.raises(KeyError):
        # Expect raised exception when a residue does not contain an atom
        strucinfo.vdw_radius_protor("ALA", "K")
    # For all other unknown radii expect None
    assert strucinfo.vdw_radius_protor("HOH", "O") is None


def test_single_radii():
    assert strucinfo.vdw_radius_single("N") == 1.55


def test_full_name():
    assert strucinfo.full_name("Ala").upper() == "ALANINE"
    assert strucinfo.full_name("ALA").upper() == "ALANINE"


def test_link_type():
    assert strucinfo.link_type("Ala").upper() == "L-PEPTIDE LINKING"
    assert strucinfo.link_type("ALA").upper() == "L-PEPTIDE LINKING"


@pytest.mark.parametrize(
    "residues, should_have_one_letter, exception_ratio",
    [
        (strucinfo.amino_acid_names(), True, 0.4),
        (strucinfo.nucleotide_names(), True, 0.4),
        (
            sorted(
                set(strucinfo.all_residues())
                - set(strucinfo.amino_acid_names())
                - set(strucinfo.nucleotide_names())
            ),
            False,
            0.01,
        ),
    ],
)
def test_one_letter_code(residues, should_have_one_letter, exception_ratio):
    """
    Generally, expect that for amino acids and nucleotides the one-letter code
    is defined, while for the rest it is not.
    However, there are a relatively high number of exceptions.
    """
    exception_count = 0
    for res_name in residues:
        has_one_letter = strucinfo.one_letter_code(res_name) is not None
        if should_have_one_letter ^ has_one_letter:
            exception_count += 1
    assert exception_count / len(residues) <= exception_ratio


@pytest.mark.parametrize(
    "multi_model, seed", itertools.product([False, True], range(10))
)
def test_standardize_order(multi_model, seed):
    original = load_structure(join(data_dir("structure"), "1l2y.bcif"))
    if not multi_model:
        original = original[0]
    # The box is not preserved when concatenating atom arrays later
    # This would complicate the atom array equality later
    original.box = None

    # Randomly reorder the atoms in each residue
    np.random.seed(seed)
    if multi_model:
        reordered = struc.AtomArrayStack(original.stack_depth(), 0)
    else:
        reordered = struc.AtomArray(0)
    for residue in struc.residue_iter(original):
        bound = residue.array_length()
        indices = np.random.choice(np.arange(bound), bound, replace=False)
        reordered += residue[..., indices]

    # Restore the original PDB standard order
    restored = reordered[..., strucinfo.standardize_order(reordered)]

    assert restored.shape == original.shape
    assert (
        restored[..., restored.element != "H"] == original[..., original.element != "H"]
    )


def test_set_ccd_path(fake_ccd_path):
    """
    Test if the CCD path can be set and the CCD is loaded correctly from it.
    """
    # Access CCD before setting it to a new path to check if the cache is cleared
    strucinfo.all_residues()

    strucinfo.set_ccd_path(fake_ccd_path)

    # The new fake CCD has only a single compound
    assert strucinfo.all_residues() == ["FOO"]
