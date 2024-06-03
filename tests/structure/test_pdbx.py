# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import warnings
import glob
import itertools
from os.path import join, splitext
import numpy as np
import pytest
from pytest import approx
import biotite
import biotite.sequence as seq
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from ..util import data_dir


@pytest.mark.parametrize("format", ["cif", "bcif", "legacy"])
def test_get_model_count(format):
    """
    Check of :func:`get_model_count()`gives the same number of models
    as :func:`get_structure()`.
    """
    base_path = join(data_dir("structure"), f"1l2y")
    if format == "cif":
        pdbx_file = pdbx.CIFFile.read(base_path + ".cif")
    elif format == "bcif":
        pdbx_file = pdbx.BinaryCIFFile.read(base_path + ".bcif")
    else:
        pdbx_file = pdbx.PDBxFile.read(base_path + ".cif")
    test_model_count = pdbx.get_model_count(pdbx_file)
    ref_model_count = pdbx.get_structure(pdbx_file).stack_depth()
    assert test_model_count == ref_model_count


@pytest.mark.parametrize(
    "string, looped",
    itertools.product(
        ["", " ", "  ", "te  xt", "'", '"' ,"te\nxt", "\t",],
        [False, True]
    ),
)
def test_escape(string, looped):
    """
    Test whether values that need to be escaped are properly escaped.
    The confirmation is done by deserialize the serialized value.
    """
    LENGTH = 5
    ref_value = np.full(LENGTH, string).tolist() if looped else string
    ref_category = pdbx.CIFCategory({"test_col": ref_value}, "test_cat")

    test_category = pdbx.CIFCategory.deserialize(ref_category.serialize())
    if looped:
        test_value = test_category["test_col"].as_array(str).tolist()
    else:
        test_value = test_category["test_col"].as_item()

    assert test_value == ref_value


@pytest.mark.parametrize(
    "format, path, model",
    itertools.product(
        ["cif", "bcif", "legacy"],
        glob.glob(join(data_dir("structure"), "*.cif")),
        [None, 1, -1]
    ),
)
def test_conversion(tmpdir, format, path, model):
    """
    Test serializing and deserializing a structure from a file
    restores the same structure.
    """
    DELETED_ANNOTATION = "auth_comp_id"

    base_path = splitext(path)[0]
    if format == "cif":
        data_path = base_path + ".cif"
        File = pdbx.CIFFile
    elif format == "bcif":
        data_path = base_path + ".bcif"
        File = pdbx.BinaryCIFFile
    else:
        data_path = base_path + ".cif"
        File = pdbx.PDBxFile

    pdbx_file = File.read(data_path)
    try:
        ref_atoms = pdbx.get_structure(
            pdbx_file, model=model, include_bonds=True
        )
    except biotite.InvalidFileError:
        if model is None:
            # The file cannot be parsed into an AtomArrayStack,
            # as the models contain different numbers of atoms
            # -> skip this test case
            return
        else:
            raise

    pdbx_file = File()
    pdbx.set_structure(pdbx_file, ref_atoms)
    file_path = join(tmpdir, f"test.{format}")
    pdbx_file.write(file_path)

    pdbx_file = File.read(file_path)
    # Remove one label section to test fallback to auth fields
    if format == "legacy":
        del pdbx_file.cif_file.block["atom_site"][DELETED_ANNOTATION]
    else:
        del pdbx_file.block["atom_site"][DELETED_ANNOTATION]
    with pytest.warns(UserWarning, match=f"'{DELETED_ANNOTATION}' not found"):
        test_atoms = pdbx.get_structure(
            pdbx_file, model=model, include_bonds=True
        )

    assert ref_atoms.array_length() > 0
    if ref_atoms.box is not None:
        assert np.allclose(test_atoms.box, test_atoms.box)
    assert test_atoms.bonds == ref_atoms.bonds
    for category in ref_atoms.get_annotation_categories():
        assert (
            test_atoms.get_annotation(category).tolist()
            == ref_atoms.get_annotation(category).tolist()
        )
    assert test_atoms.coord.tolist() == ref_atoms.coord.tolist()


@pytest.mark.parametrize(
    "format, path",
    itertools.product(
        ["cif", "bcif"],
        glob.glob(join(data_dir("structure"), "*.cif")),
    ),
)
def test_bond_conversion(tmpdir, format, path):
    """
    Test serializing and deserializing bonds from a file
    restores the bonds.

    This test is similar to :func:`test_conversion`, but intra bonds
    are written to ``chem_comp_bond`` and read from there, instead of
    relying on the CCD.
    """
    base_path = splitext(path)[0]
    if format == "cif":
        data_path = base_path + ".cif"
        File = pdbx.CIFFile
    else:
        data_path = base_path + ".bcif"
        File = pdbx.BinaryCIFFile

    pdbx_file = File.read(data_path)
    atoms = pdbx.get_structure(
        pdbx_file, model=1, include_bonds=True
    )
    ref_bonds = atoms.bonds

    pdbx_file = File()
    # The import difference to `test_conversion()` is `include_bonds`
    pdbx.set_structure(pdbx_file, atoms, include_bonds=True)
    file_path = join(tmpdir, f"test.{format}")
    pdbx_file.write(file_path)

    pdbx_file = File.read(file_path)
    # Ensure that the CCD fallback is not used,
    # i.e. the bonds can be properly read from ``chem_comp_bond``
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        test_bonds = pdbx.get_structure(
            pdbx_file, model=1, include_bonds=True
        ).bonds

    assert test_bonds == ref_bonds


@pytest.mark.parametrize(
    "format", ["cif", "bcif"]
)
def test_extra_fields(tmpdir, format):
    path = join(data_dir("structure"), f"1l2y.{format}")
    if format == "cif":
        File = pdbx.CIFFile
    else:
        File = pdbx.BinaryCIFFile

    pdbx_file = File.read(path)
    ref_atoms = pdbx.get_structure(
        pdbx_file, extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
    )

    pdbx_file = File()
    pdbx.set_structure(pdbx_file, ref_atoms, data_block="test")
    file_path = join(tmpdir, "test")
    pdbx_file.write(file_path)

    pdbx_file = File.read(path)
    test_atoms = pdbx.get_structure(
        pdbx_file, extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
    )

    assert test_atoms.ins_code.tolist() == ref_atoms.ins_code.tolist()
    assert test_atoms.atom_id.tolist() == ref_atoms.atom_id.tolist()
    assert test_atoms.b_factor.tolist() == approx(ref_atoms.b_factor.tolist())
    assert test_atoms.occupancy.tolist() == approx(ref_atoms.occupancy.tolist())
    assert test_atoms.charge.tolist() == ref_atoms.charge.tolist()
    assert test_atoms == ref_atoms


def test_intra_bond_residue_parsing():
    """
    Check if intra-residue bonds can be parsed from a NextGen CIF file
    and expect the same bonds as the CCD-based ones from an *original*
    CIF file.
    """
    cif_path = join(data_dir("structure"), "1l2y.cif")
    cif_file = pdbx.CIFFile.read(cif_path)
    ref_bonds = pdbx.get_structure(
        cif_file, model=1, include_bonds=True
    ).bonds

    nextgen_cif_path = join(
        data_dir("structure"), "nextgen", "pdb_00001l2y_xyz-enrich.cif"
    )
    nextgen_cif_file = pdbx.CIFFile.read(nextgen_cif_path)
    # Ensure that the CCD fallback is not used,
    # i.e. the bonds can be properly read from ``chem_comp_bond``
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        test_bonds = pdbx.get_structure(
            nextgen_cif_file, model=1, include_bonds=True
        ).bonds

    assert test_bonds == ref_bonds


@pytest.mark.parametrize(
    "format", ["cif", "bcif"]
)
def test_any_bonds(tmpdir, format):
    """
    Check if ``BondType.ANY`` bonds can be written and read from a PDBx
    file, i.e. the ``chem_comp_bond`` and ``struct_conn`` categories.
    ``BondType.ANY`` is represented by missing values.
    """
    N_ATOMS = 4

    File = pdbx.CIFFile if format == "cif" else pdbx.BinaryCIFFile

    atoms = struc.AtomArray(N_ATOMS)
    atoms.coord[...] = 1.0
    atoms.atom_name = [f"C{i}" for i in range(N_ATOMS)]
    # Two different residues to test inter-residue bonds as well
    atoms.res_id = [0, 0, 1, 1]
    atoms.res_name = ["A", "A", "B", "B"]

    ref_bonds = struc.BondList(N_ATOMS)
    # Intra-residue bond
    ref_bonds.add_bond(0, 1, struc.BondType.ANY)
    # Inter-residue bond
    ref_bonds.add_bond(1, 2, struc.BondType.ANY)
    # Intra-residue bond
    ref_bonds.add_bond(2, 3, struc.BondType.ANY)
    atoms.bonds = ref_bonds

    pdbx_file = File()
    pdbx.set_structure(pdbx_file, atoms, include_bonds=True)
    file_path = join(tmpdir, f"test.{format}")
    pdbx_file.write(file_path)

    pdbx_file = File.read(file_path)
    # Ensure that the CCD fallback is not used,
    # i.e. the bonds can be properly read from ``chem_comp_bond``
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        test_bonds = pdbx.get_structure(
            pdbx_file, model=1, include_bonds=True
        ).bonds

    assert test_bonds == ref_bonds


@pytest.mark.parametrize(
    "format", ["cif", "bcif"]
)
def test_unequal_lengths(format):
    """
    Check if setting columns with unequal lengths in the same category
    raises an exception.
    """
    if format == "cif":
        Category = pdbx.CIFCategory
    else:
        Category = pdbx.BinaryCIFCategory

    # Arrays have unequal lengths -> invalid
    invalid_category_dict = {
        "foo1": np.arange(3),
        "foo2": np.arange(4),
    }
    with pytest.raises(pdbx.SerializationError):
        Category(invalid_category_dict).serialize()


def test_setting_empty_column():
    """
    Check if setting an empty column raises an exception.
    """
    with pytest.raises(
        ValueError, match="Array must contain at least one element"
    ):
        pdbx.CIFCategory({"foo": []})


def test_setting_empty_structure():
    """
    Check if setting an empty structure raises an exception.
    In contrast, empty bonds should be simply ignored
    """
    empty_atoms = struc.AtomArray(0)
    with pytest.raises(struc.BadStructureError):
        pdbx.set_structure(pdbx.CIFFile(), empty_atoms)

    atoms = struc.AtomArray(1)
    # Residue and atom name are required for setting intra-residue bonds
    atoms.res_name[:] = "ALA"
    atoms.atom_name[:] = "A"
    atoms.coord[:, :] = 0.0
    empty_bonds = struc.BondList(atoms.array_length())
    atoms.bonds = empty_bonds
    pdbx.set_structure(pdbx.CIFFile(), atoms, include_bonds=True)


@pytest.mark.parametrize(
    "format", ["cif", "bcif"]
)
def test_list_assemblies(format):
    """
    Test the :func:`list_assemblies()` function based on a known
    example.
    """
    path = join(data_dir("structure"), f"1f2n.{format}")
    if format == "cif":
        File = pdbx.CIFFile
    else:
        File = pdbx.BinaryCIFFile

    pdbx_file = File.read(path)

    assembly_list = pdbx.list_assemblies(pdbx_file)
    assert assembly_list == {
        "1": "complete icosahedral assembly",
        "2": "icosahedral asymmetric unit",
        "3": "icosahedral pentamer",
        "4": "icosahedral 23 hexamer",
        "5": "icosahedral asymmetric unit, std point frame",
        "6": "crystal asymmetric unit, crystal frame",
    }


@pytest.mark.parametrize("format, pdb_id, model", itertools.product(
    ["cif", "bcif"],
    ["1f2n", "5zng"],
    [None, 1, -1]
))
def test_get_assembly(format, pdb_id, model):
    """
    Test whether the :func:`get_assembly()` function produces the same
    number of peptide chains as the
    ``_pdbx_struct_assembly.oligomeric_count`` field indicates.
    Furthermore, check if the number of atoms in the entire assembly
    is a multiple of the numbers of atoms in a monomer.
    """
    path = join(data_dir("structure"), f"{pdb_id}.{format}")
    if format == "cif":
        File = pdbx.CIFFile
    else:
        File = pdbx.BinaryCIFFile

    pdbx_file = File.read(path)

    assembly_category = pdbx_file.block["pdbx_struct_assembly"]
    # Test each available assembly
    for id, ref_oligomer_count in zip(
        assembly_category["id"].as_array(str),
        assembly_category["oligomeric_count"].as_array(int)
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
        assert test_oligomer_count == ref_oligomer_count

        # The atom count of the entire assembly should be a multiple
        # a monomer,
        monomer_atom_count = pdbx.get_structure(pdbx_file).array_length()
        assert assembly.array_length() % monomer_atom_count == 0


@pytest.mark.parametrize(
    "path, use_ideal_coord",
    itertools.product(
        glob.glob(join(data_dir("structure"), "molecules", "*.cif")),
        [False, True]
    ),
)
def test_component_conversion(tmpdir, path, use_ideal_coord):
    """
    After reading a component from a CIF file, writing the component
    back to a new file and reading it again should give the same
    structure.
    """
    cif_file = pdbx.CIFFile.read(path)
    ref_atoms = pdbx.get_component(
        cif_file, use_ideal_coord=use_ideal_coord
    )

    cif_file = pdbx.CIFFile()
    pdbx.set_component(cif_file, ref_atoms, data_block="test")
    file_path = join(tmpdir, "test")
    cif_file.write(file_path)

    cif_file = pdbx.CIFFile.read(path)
    test_atoms = pdbx.get_component(
        cif_file, use_ideal_coord=use_ideal_coord
    )

    assert test_atoms == ref_atoms


@pytest.mark.parametrize("format", ["cif", "bcif"])
def test_get_sequence(format):
    """
    Check if the :func:`get_sequence()` function returns the correct
    sequences for a known example.
    """
    if format == "cif":
        File = pdbx.CIFFile
    else:
        File = pdbx.BinaryCIFFile

    pdbx_file = File.read(join(data_dir("structure"), f"5ugo.{format}"))
    sequences = pdbx.get_sequence(pdbx_file)
    pdbx_file = File.read(join(data_dir("structure"), f"4gxy.{format}"))
    sequences += pdbx.get_sequence(pdbx_file)
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


def test_bcif_encoding():
    """
    Check if encoding and subsequent decoding data in a BinaryCIF file
    restores the original data.
    """
    PDB_ID = "1aki"

    encodings_used = {
        encoding: False for encoding in [
            pdbx.ByteArrayEncoding,
            pdbx.FixedPointEncoding,
            # This encoding is not used in the test file
            #pdbx.IntervalQuantizationEncoding,
            pdbx.RunLengthEncoding,
            pdbx.DeltaEncoding,
            pdbx.IntegerPackingEncoding,
            pdbx.StringArrayEncoding
        ]
    }

    bcif_file = pdbx.BinaryCIFFile.read(
        join(data_dir("structure"), f"{PDB_ID}.bcif")
    )
    for category_name, category in bcif_file[PDB_ID.upper()].items():
        for column_name in category.keys():
            try:
                ref_msgpack = category._elements[column_name]
                # Remove the identifier, since it is not created in the
                # serialization
                # (This would be added when the category is serialized)
                del ref_msgpack["name"]

                column = category[column_name]

                for enc in column.data.encoding:
                    encodings_used[type(enc)] = True
                if column.mask is not None:
                    for enc in column.mask.encoding:
                        encodings_used[type(enc)] = True
                test_msgpack = column.serialize()

                assert test_msgpack == ref_msgpack
            except:
                raise Exception(
                    f"Encoding failed for '{category_name}.{column_name}'"
                )

    # Check if each encoding was used at least once
    # to ensure that the test was thorough
    for key, was_used in encodings_used.items():
        try:
            assert was_used
        except:
            raise Exception(f"Encoding {key} was not used")


def test_bcif_quantization_encoding():
    """
    As the :class:`IntervalQuantizationEncoding` is not used in the test
    file, this test checks if it is capable of encoding and
    decoding artificial data.
    """
    MIN = 5
    MAX = 42
    NUM_STEPS = 100

    np.random.seed(0)
    ref_data = np.linspace(MIN, MAX, NUM_STEPS)
    np.random.shuffle(ref_data)

    encoding = pdbx.IntervalQuantizationEncoding(MIN, MAX, NUM_STEPS)
    test_data = encoding.decode(encoding.encode(ref_data))

    assert test_data.tolist() == approx(ref_data.tolist())


def test_bcif_cif_consistency():
    """
    Check if the decoded data from a BinaryCIF file is consistent with
    the equivalent data from the original CIF file.
    """
    BLACKLIST = [
        # Has value '15.' in CIF,
        # which cannot be converted to into of BCIF
        ("reflns", "d_resolution_low")
    ]
    PDB_ID = "1aki"

    base_path = join(data_dir("structure"), PDB_ID)
    cif_file = pdbx.CIFFile.read(base_path + ".cif")
    bcif_file = pdbx.BinaryCIFFile.read(base_path + ".bcif")

    cif_block = cif_file.block
    bcif_block = bcif_file.block
    assert set(cif_block.keys()) == set(bcif_block.keys())

    for category_name in cif_block.keys():
        cif_category = cif_block[category_name]
        bcif_category = bcif_block[category_name]
        assert set(cif_category.keys()) == set(bcif_category.keys())

        for column_name in cif_category.keys():
            if (category_name, column_name) in BLACKLIST:
                continue
            try:
                cif_column = cif_category[column_name]
                bcif_column = bcif_category[column_name]
                if cif_column.mask is None:
                    assert bcif_column.mask is None
                else:
                    assert cif_column.mask.array.tolist() \
                        == bcif_column.mask.array.tolist()
                # In CIF format, all vales are strings
                # -> ensure consistency
                dtype = bcif_column.data.array.dtype
                assert cif_column.as_array(dtype).tolist() \
                    == pytest.approx(bcif_column.as_array(dtype).tolist())
            except:
                raise Exception(
                    f"Comparison failed for '{category_name}.{column_name}'"
                )


@pytest.mark.parametrize(
    "format, create_new_encoding",
    [
        ("cif", None),
        ("bcif", False),
        ("bcif", True),
    ]
)
def test_serialization_consistency(format, create_new_encoding):
    """
    Check if deserializing data, that was serialized before, is still
    the same.

    ``create_new_encoding=True`` tests if the default encodings created
    by :class:`BinaryCIFFile` successfully encode and decode the data.
    """
    PDB_ID = "1aki"

    path = join(data_dir("structure"), f"{PDB_ID}.{format}")
    if format == "cif":
        file = pdbx.CIFFile.read(path)
    elif format == "bcif":
        file = pdbx.BinaryCIFFile.read(path)

    for category_name, ref_category in file.block.items():
        if format == "cif":
            test_category = pdbx.CIFCategory.deserialize(
                ref_category.serialize()
            )
        elif format == "bcif":
            # Access each column to force otherwise lazy deserialization
            for _ in ref_category.values():
                pass
            if create_new_encoding:
                ref_category = _clear_encoding(ref_category)
            test_category = pdbx.BinaryCIFCategory.deserialize(
                ref_category.serialize()
            )
        try:
            for key in test_category.keys():
                assert ref_category[key] == test_category[key]
        except:
            raise Exception(f"Comparison failed for '{category_name}.{key}'")


def test_legacy_pdbx():
    PDB_ID = "1aki"

    path = join(data_dir("structure"), f"{PDB_ID}.cif")
    ref_file = pdbx.CIFFile.read(path)

    test_file = pdbx.PDBxFile.read(path)
    assert test_file.get_block_names() == [PDB_ID.upper()]

    for category_name, category in ref_file.block.items():
        test_category_dict = test_file.get_category(
            category_name, PDB_ID.upper(), expect_looped=True
        )
        for column_name, test_array in test_category_dict.items():
            try:
                assert test_array.tolist() \
                    == category[column_name].as_array(str).tolist()
            except:
                raise Exception(
                    f"Comparison failed for {category_name}.{column_name}"
                )


def _clear_encoding(category):
    columns = {}
    for key, col in category.items():
        # Clearing is done by not specifying an encoding
        # in the BinaryCIFData creation
        data = pdbx.BinaryCIFData(col.data.array)
        if col.mask is not None:
            mask = pdbx.BinaryCIFData(col.mask.array)
        else:
            mask = None
        columns[key] = pdbx.BinaryCIFColumn(data, mask)
    return pdbx.BinaryCIFCategory(columns)
