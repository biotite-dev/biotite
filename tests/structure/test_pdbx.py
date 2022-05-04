# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
import itertools
from os.path import join
import numpy as np
import pytest
from pytest import approx
import biotite
import biotite.sequence as seq
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from ..util import data_dir


def test_get_model_count():
    pdbx_file = pdbx.PDBxFile.read(join(data_dir("structure"), "1l2y.cif"))
    test_model_count = pdbx.get_model_count(pdbx_file)
    ref_model_count = pdbx.get_structure(pdbx_file).stack_depth()
    assert test_model_count == ref_model_count


@pytest.mark.parametrize(
    "category, key, exp_value",
    [
        (
            "audit_author",
            "name",
            ["Neidigh, J.W.", "Fesinmeyer, R.M.", "Andersen, N.H."],
        ),
        ("struct_ref_seq", "pdbx_PDB_id_code", "1L2Y"),
        (
            "pdbx_nmr_ensemble",
            "conformer_selection_criteria",
            "structures with acceptable covalent geometry, "
            "structures with the least restraint violations",
        ),
    ],
)
def test_parsing(category, key, exp_value):
    pdbx_file = pdbx.PDBxFile.read(join(data_dir("structure"), "1l2y.cif"))
    cat_dict = pdbx_file[category]
    value = cat_dict[key]
    if isinstance(value, np.ndarray):
        assert value.tolist() == exp_value
    else:
        assert value == exp_value


@pytest.mark.parametrize(
    "string, use_array",
    itertools.product(["", " ", "\n", "\t"], [False, True]),
)
def test_empty_values(string, use_array):
    """
    Test whether empty strings for field values are properly replaced
    by ``'.'``.
    """
    LENGTH = 10
    ref_value = np.full(LENGTH, string, dtype="U1") if use_array else ""
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.set_category(
        category="test_category",
        block="test",
        category_dict={"test_field": ref_value},
    )

    test_value = pdbx_file["test_category"]["test_field"]

    if use_array:
        assert test_value.tolist() == ["."] * LENGTH
    else:
        assert test_value == "."


@pytest.mark.parametrize(
    "path, model",
    itertools.product(
        glob.glob(join(data_dir("structure"), "*.cif")), [None, 1, -1]
    ),
)
def test_conversion(path, model):
    pdbx_file = pdbx.PDBxFile.read(path)

    try:
        array1 = pdbx.get_structure(pdbx_file, model=model)
    except biotite.InvalidFileError:
        if model is None:
            # The file cannot be parsed into an AtomArrayStack,
            # as the models contain different numbers of atoms
            # -> skip this test case
            return
        else:
            raise

    pdbx_file = pdbx.PDBxFile()
    pdbx.set_structure(pdbx_file, array1, data_block="test")

    # Remove one optional auth section in label to test fallback to label
    # fields
    atom_cat = pdbx_file.get_category("atom_site", "test")
    atom_cat.pop("auth_atom_id")
    pdbx_file.set_category("atom_site", atom_cat, "test")

    array2 = pdbx.get_structure(pdbx_file, model=model)

    assert array1.array_length() > 0
    if array1.box is not None:
        assert np.allclose(array1.box, array2.box)
    assert array1.bonds == array2.bonds
    for category in array1.get_annotation_categories():
        assert (
            array1.get_annotation(category).tolist()
            == array2.get_annotation(category).tolist()
        )
    assert array1.coord.tolist() == array2.coord.tolist()


def test_extra_fields():
    path = join(data_dir("structure"), "1l2y.cif")
    pdbx_file = pdbx.PDBxFile.read(path)
    stack1 = pdbx.get_structure(
        pdbx_file, extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
    )
    pdbx_file = pdbx.PDBxFile()
    pdbx.set_structure(pdbx_file, stack1, data_block="test")
    stack2 = pdbx.get_structure(
        pdbx_file, extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
    )
    assert stack1 == stack2

    path = join(data_dir("structure"), "1l2y.cif")
    pdbx_file = pdbx.PDBxFile.read(path)
    stack1 = pdbx.get_structure(
        pdbx_file, extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
    )

    pdbx_file = pdbx.PDBxFile()
    pdbx.set_structure(pdbx_file, stack1, data_block="test")

    stack2 = pdbx.get_structure(
        pdbx_file, extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
    )

    assert stack1.ins_code.tolist() == stack2.ins_code.tolist()
    assert stack1.atom_id.tolist() == stack2.atom_id.tolist()
    assert stack1.b_factor.tolist() == approx(stack2.b_factor.tolist())
    assert stack1.occupancy.tolist() == approx(stack2.occupancy.tolist())
    assert stack1.charge.tolist() == stack2.charge.tolist()
    assert stack1 == stack2


def test_unequal_lengths():
    valid_category_dict = {"foo1": ["1", "2", "3"], "foo2": ["1", "2", "3"]}
    # Arrays have unequal lengths -> invalid
    invalid_category_dict = {
        "foo1": ["1", "2", "3"],
        "foo2": ["1", "2", "3", "4"],
    }
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.set_category("test", valid_category_dict, block="test_block")
    with pytest.raises(ValueError):
        pdbx_file.set_category(
            "test", invalid_category_dict, block="test_block"
        )


def test_list_assemblies():
    """
    Test the :func:`list_assemblies()` function based on a known
    example.
    """
    path = join(data_dir("structure"), "1f2n.cif")
    pdbx_file = pdbx.PDBxFile.read(path)

    assembly_list = pdbx.list_assemblies(pdbx_file)
    assert assembly_list == {
        "1": "complete icosahedral assembly",
        "2": "icosahedral asymmetric unit",
        "3": "icosahedral pentamer",
        "4": "icosahedral 23 hexamer",
        "5": "icosahedral asymmetric unit, std point frame",
        "6": "crystal asymmetric unit, crystal frame",
    }


@pytest.mark.parametrize("pdb_id, model", itertools.product(
    ["1f2n", "5zng"],
    [None, 1, -1]
))
def test_get_assembly(pdb_id, model):
    """
    Test whether the :func:`get_assembly()` function produces the same
    number of peptide chains as the
    ``_pdbx_struct_assembly.oligomeric_count`` field indicates.
    Furthermore, check if the number of atoms in the entire assembly
    is a multiple of the numbers of atoms in a monomer.
    """

    path = join(data_dir("structure"), f"{pdb_id}.cif")
    pdbx_file = pdbx.PDBxFile.read(path)

    assembly_category = pdbx_file.get_category(
        "pdbx_struct_assembly", expect_looped=True
    )
    # Test each available assembly
    for id, ref_oligomer_count in zip(
        assembly_category["id"], assembly_category["oligomeric_count"]
    ):
        print("Assembly ID:", id)
        try:
            assembly = pdbx.get_assembly(
                pdbx_file, assembly_id=id, model=model
            )
        except biotite.InvalidFileError:
            if model is None:
                # The file cannot be parsed into an AtomArrayStack,
                # as the models contain different numbers of atoms
                # -> skip this test case
                return
            else:
                raise
        protein_assembly = assembly[..., struc.filter_amino_acids(assembly)]
        test_oligomer_count = struc.get_chain_count(protein_assembly)

        if model is None:
            assert isinstance(assembly, struc.AtomArrayStack)
        else:
            assert isinstance(assembly, struc.AtomArray)
        assert test_oligomer_count == int(ref_oligomer_count)

        # The atom count of the entire assembly should be a multiple
        # a monomer,
        monomer_atom_count = pdbx.get_structure(pdbx_file).array_length()
        assert assembly.array_length() % monomer_atom_count == 0


def test_get_sequence():
    file = pdbx.PDBxFile.read(join(data_dir("structure"), "5ugo.cif"))
    sequences = pdbx.get_sequence(file)
    file = pdbx.PDBxFile.read(join(data_dir("structure"), "4gxy.cif"))
    sequences += pdbx.get_sequence(file)
    assert str(sequences[0]) == "CCGACGGCGCATCAGC"
    assert type(sequences[0]) is seq.NucleotideSequence
    assert str(sequences[1]) == "GCTGATGCGCC"
    assert type(sequences[1]) is seq.NucleotideSequence
    assert str(sequences[2]) == "GTCGG"
    assert type(sequences[2]) is seq.NucleotideSequence
    assert (
        str(sequences[3]) == "MSKRKAPQETLNGGITDMLTELANFEKNVSQAIHKYN"
        "AYRKAASVIAKYPHKIKSGAEAKKLPGVGTKIAEKIDEFLATGKLRKLEKIRQD"
        "DTSSSINFLTRVSGIGPSAARKFVDEGIKTLEDLRKNEDKLNHHQRIGLKYFGD"
        "FEKRIPREEMLQMQDIVLNEVKKVDSEYIATVCGSFRRGAESSGDMDVLLTHPS"
        "FTSESTKQPKLLHQVVEQLQKVHFITDTLSKGETKFMGVCQLPSKNDEKEYPHR"
        "RIDIRLIPKDQYYCGVLYFTGSDIFNKNMRAHALEKGFTINEYTIRPLGVTGVA"
        "GEPLPVDSEKDIFDYIQWKYREPKDRSE"
    )
    assert type(sequences[3]) is seq.ProteinSequence
    assert (
        str(sequences[4]) == "GGCGGCAGGTGCTCCCGACCCTGCGGTCGGGAGTTAA"
        "AAGGGAAGCCGGTGCAAGTCCGGCACGGTCCCGCCACTGTGACGGGGAGTCGCC"
        "CCTCGGGATGTGCCACTGGCCCGAAGGCCGGGAAGGCGGAGGGGCGGCGAGGAT"
        "CCGGAGTCAGGAAACCTGCCTGCCGTC"
    )
    assert type(sequences[4]) is seq.NucleotideSequence
