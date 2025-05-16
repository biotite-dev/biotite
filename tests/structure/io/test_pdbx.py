# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
import itertools
import warnings
from io import BytesIO
from os.path import join, splitext
from pathlib import Path
import msgpack
import numpy as np
import pytest
from pytest import approx
import biotite
import biotite.sequence as seq
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from biotite.structure.io.pdbx.bcif import _encode_numpy as encode_numpy
from biotite.structure.io.pdbx.compress import _get_decimal_places as get_decimal_places
from tests.util import data_dir


@pytest.mark.parametrize("format", ["cif", "bcif"])
def test_get_model_count(format):
    """
    Check of :func:`get_model_count()`gives the same number of models
    as :func:`get_structure()`.
    """
    base_path = join(data_dir("structure"), "1l2y")
    if format == "cif":
        pdbx_file = pdbx.CIFFile.read(base_path + ".cif")
    else:
        pdbx_file = pdbx.BinaryCIFFile.read(base_path + ".bcif")
    test_model_count = pdbx.get_model_count(pdbx_file)
    ref_model_count = pdbx.get_structure(pdbx_file).stack_depth()
    assert test_model_count == ref_model_count


@pytest.mark.parametrize(
    "string, looped",
    itertools.product(
        [
            "",
            " ",
            "  ",
            "te  xt",
            "'",
            '"',
            "te\nxt",
            "\t",
            """single"anddouble"marks""",
            """single' and double" marks with whitespace""",
        ],
        [False, True],
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
    "cif_line, expected_fields",
    [
        ["'' 'embed'quote' ", ["", "embed'quote"]],
        ['2 "embed"quote" "\t\n"', ["2", 'embed"quote', "\t\n"]],
        [" 3 '' \"\" 'spac e' 'embed\"quote'", ["3", "", "", "spac e", 'embed"quote']],
        ["''' \"\"\" ''quoted''", ["'", '"', "'quoted'"]],
    ],
)
def test_split_one_line(cif_line, expected_fields):
    """
    Test whether values that have an embedded quote are properly escaped.
    """
    assert list(pdbx.cif._split_one_line(cif_line)) == expected_fields


@pytest.mark.parametrize("find_matches_by_dict", [False, True])
@pytest.mark.parametrize("model", [None, 1, -1])
@pytest.mark.parametrize(
    "path", Path(data_dir("structure")).glob("*.cif"), ids=lambda p: p.stem
)
@pytest.mark.parametrize("format", ["cif", "bcif"])
def test_conversion(monkeypatch, tmpdir, format, path, model, find_matches_by_dict):
    """
    Test serializing and deserializing a structure from a file
    restores the same structure.
    """
    DELETED_ANNOTATION = "auth_comp_id"

    if find_matches_by_dict:
        # Lower the threshold to 0 to force usage of `_find_matches_by_dict()`
        monkeypatch.setattr(pdbx.convert, "FIND_MATCHES_SWITCH_THRESHOLD", 0)

    base_path = splitext(path)[0]
    if format == "cif":
        data_path = base_path + ".cif"
        File = pdbx.CIFFile
    else:
        data_path = base_path + ".bcif"
        File = pdbx.BinaryCIFFile

    pdbx_file = File.read(data_path)
    try:
        ref_atoms = pdbx.get_structure(pdbx_file, model=model, include_bonds=True)
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
    del pdbx_file.block["atom_site"][DELETED_ANNOTATION]
    with pytest.warns(UserWarning, match=f"'{DELETED_ANNOTATION}' not found"):
        test_atoms = pdbx.get_structure(pdbx_file, model=model, include_bonds=True)

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
    "pdb_id",
    [
        "1aki",  # has no altloc IDs
        "3o5r",  # has altloc IDs
    ],
)
def test_filter_altloc(pdb_id):
    """
    Check if the different ``altloc`` options give the expected results.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), f"{pdb_id}.bcif"))
    atoms = {}
    for altloc in ["first", "occupancy", "all"]:
        atoms[altloc] = pdbx.get_structure(pdbx_file, model=1, altloc=altloc)

    # The 'altloc_id' annotation should only be present in the 'altloc=all' case
    assert "altloc_id" in atoms["all"].get_annotation_categories()
    assert "altloc_id" not in atoms["first"].get_annotation_categories()
    assert "altloc_id" not in atoms["occupancy"].get_annotation_categories()
    # Independent of which altloc atom is selected, only one atom must be selected...
    assert atoms["occupancy"].array_length() == atoms["first"].array_length()
    # ...with the exception of the 'altloc=all' case in which more atoms are selected
    if np.any(atoms["all"].altloc_id != "."):
        assert atoms["all"].array_length() > atoms["first"].array_length()
    else:
        assert atoms["all"].array_length() == atoms["first"].array_length()


@pytest.mark.parametrize("format", ["cif", "bcif"])
def test_bonds_from_ccd(format):
    """
    Check if bonds can also be correctly restored from a CIF file that does not contain
    explicit bond information (i.e. the `chem_comp_bond` and `struct_conn` categories)
    using the CCD.

    Importantly a structure is chosen that does not contain non-standard inter-residue
    bonds, such as disulfide bridges or glycosylations, as those bonds would require
    explicit bond information.
    """
    path = join(data_dir("structure"), f"1l2y.{format}")
    File = pdbx.CIFFile if format == "cif" else pdbx.BinaryCIFFile

    pdbx_file = File.read(path)
    atoms = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    ref_bonds = atoms.bonds

    del pdbx_file.block["chem_comp_bond"]
    test_bonds = pdbx.get_structure(pdbx_file, model=1, include_bonds=True).bonds

    assert test_bonds == ref_bonds


def test_metal_coordination_bonds():
    """
    Test if metal coordination bonds are properly written and read,
    based on an known example.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), "1crr.bcif"))
    atoms = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)

    bond_array = atoms.bonds.as_array()
    # Filter bonds to bonds containing the magnesium atom
    metal_bond_array = bond_array[
        (atoms.res_name[bond_array[:, 0]] == "MG")
        | (atoms.res_name[bond_array[:, 1]] == "MG")
    ]
    assert np.all(metal_bond_array[:, 2] == struc.BondType.COORDINATION)

    # Only keep the metal bonds
    atoms.bonds = struc.BondList(atoms.array_length(), metal_bond_array)
    pdbx_file = pdbx.BinaryCIFFile()
    pdbx.set_structure(pdbx_file, atoms)
    conn_type_id = pdbx_file.block["struct_conn"]["conn_type_id"].as_array(str)
    # All inter-residue bonds in the chosen structure are metal bonds
    assert len(conn_type_id) == len(metal_bond_array)
    assert np.all(conn_type_id == "metalc")


def test_bond_sparsity():
    """
    Ensure that only as much intra-residue bonds are written as necessary,
    i.e. the created ``chem_comp_bond`` category has at maximum category many rows as
    the reference PDBx file.

    Less bonds are allowed, as not all atoms that a residue has in the CCD are actually
    present in the structure.

    This tests a previous bug, where duplicate intra-residue bonds were written
    (https://github.com/biotite-dev/biotite/issues/652).
    """
    path = join(data_dir("structure"), "1l2y.bcif")
    ref_pdbx_file = pdbx.BinaryCIFFile.read(path)
    ref_bond_number = ref_pdbx_file.block["chem_comp_bond"].row_count

    atoms = pdbx.get_structure(ref_pdbx_file, model=1, include_bonds=True)
    test_pdbx_file = pdbx.BinaryCIFFile()
    pdbx.set_structure(test_pdbx_file, atoms)
    test_bond_number = test_pdbx_file.block["chem_comp_bond"].row_count

    assert test_bond_number <= ref_bond_number


@pytest.mark.parametrize("format", ["cif", "bcif"])
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


def test_dynamic_dtype():
    """
    Check if the dtype of an annotation array is automatically adjusted if the
    column in the `atom_site` category contains longer strings than supported by the
    default dtype.
    """
    CHAIN_ID = "LONG_ID"

    path = join(data_dir("structure"), "1l2y.bcif")
    pdbx_file = pdbx.BinaryCIFFile.read(path)
    category = pdbx_file.block["atom_site"]
    category["label_asym_id"] = np.full(len(category["label_asym_id"]), CHAIN_ID)
    atoms = pdbx.get_structure(pdbx_file, model=1, use_author_fields=False)

    # Without a dynamically chosen compatible dtype, the string would be truncated
    assert (atoms.chain_id == CHAIN_ID).all()


@pytest.mark.parametrize("format", ["cif", "bcif"])
def test_any_bonds(tmpdir, format):
    """
    Check if ``BondType.ANY`` bonds can be written and read from a PDBx
    file, i.e. the ``chem_comp_bond`` and ``struct_conn`` categories.
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
    pdbx.set_structure(pdbx_file, atoms)
    file_path = join(tmpdir, f"test.{format}")
    pdbx_file.write(file_path)

    pdbx_file = File.read(file_path)
    # Ensure that the CCD fallback is not used,
    # i.e. the bonds can be properly read from ``chem_comp_bond``
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        test_bonds = pdbx.get_structure(pdbx_file, model=1, include_bonds=True).bonds
    test_bonds.remove_bond_order()

    assert test_bonds == ref_bonds


@pytest.mark.parametrize("format", ["cif", "bcif"])
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
    with pytest.raises(biotite.SerializationError):
        Category(invalid_category_dict).serialize()


def test_setting_empty_column():
    """
    Check if setting an empty column raises an exception.
    """
    with pytest.raises(ValueError, match="Array must contain at least one element"):
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
    pdbx.set_structure(pdbx.CIFFile(), atoms)


@pytest.mark.parametrize("format", ["cif", "bcif"])
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


@pytest.mark.parametrize(
    "format, pdb_id, model",
    itertools.product(["cif", "bcif"], ["1f2n", "5zng"], [None, 1, -1]),
)
def test_assembly_chain_count(format, pdb_id, model):
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
        assembly_category["oligomeric_count"].as_array(int),
    ):
        print("Assembly ID:", id)
        try:
            assembly = pdbx.get_assembly(pdbx_file, assembly_id=id, model=model)
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
    "pdb_id, assembly_id, symmetric_unit_count",
    [
        # Single operation
        ("5zng", "1", 1),
        # Multiple operations with continuous operation IDs
        ("1f2n", "1", 60),
        # Multiple operations with discontinuous operation IDs
        ("1f2n", "4", 6),
        # Multiple combined operations
        ("1f2n", "6", 60),
        # Multiple entries in pdbx_struct_assembly_gen for the same assembly_id
        ("4zxb", "1", 2),
        # Multiple entries in pdbx_struct_assembly_gen for the same assembly_id and chain
        ("1ncb", "2", 8),
    ],
)
def test_assembly_sym_id(pdb_id, assembly_id, symmetric_unit_count):
    """
    Check if the :func:`get_assembly()` function returns the correct
    number of symmetry IDs for a known example.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), f"{pdb_id}.bcif"))
    assembly = pdbx.get_assembly(pdbx_file, assembly_id=assembly_id)
    assert sorted(np.unique(assembly.sym_id).tolist()) == list(
        range(symmetric_unit_count)
    )


@pytest.mark.parametrize("model", [None, 1])
@pytest.mark.parametrize("center", [False, True])
def test_unit_cell_trivial(model, center):
    """
    The 'P 1' space group has no symmetries.
    Hence the unit cell from this space group should be the same as the asymmetric unit.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), "1l2y.bcif"))
    # Give the structure the 'P 1' space group
    pdbx_file.block["symmetry"] = pdbx.BinaryCIFCategory(
        {"space_group_name_H-M": "P 1"}
    )
    # Give the structure arbitrary unit cell dimensions,
    # it needs only be large enough to contain the asymmetric unit
    pdbx_file.block["cell"] = pdbx.BinaryCIFCategory(
        {
            "length_a": 10, "length_b": 20, "length_c": 30,
            "angle_alpha": 90, "angle_beta": 90, "angle_gamma": 90,
        }
    )  # fmt: skip
    if center:
        # Ensure that the asymmetric unit is already inside the unit cell,
        # as the centroid may sometimes be slightly outside
        for col_name in ["Cartn_x", "Cartn_y", "Cartn_z"]:
            pdbx_file.block["atom_site"][col_name] = (
                pdbx_file.block["atom_site"][col_name].as_array(np.float32) + 1.0
            )

    asymmetric_unit = pdbx.get_structure(pdbx_file, model)
    unit_cell = pdbx.get_unit_cell(pdbx_file, center, model)

    for category in asymmetric_unit.get_annotation_categories():
        assert (
            unit_cell.get_annotation(category).tolist()
            == asymmetric_unit.get_annotation(category).tolist()
        )
    assert unit_cell.coord.flatten().tolist() == approx(
        asymmetric_unit.coord.flatten().tolist()
    )


@pytest.mark.parametrize(
    "pdb_path", Path(data_dir("structure")).glob("*.pdb"), ids=lambda p: p.stem
)
def test_unit_cell_pdb_consistency(pdb_path):
    """
    Check the structure parsed via :func:`pdbx.get_unit_cell()` against
    :func:`pdb.get_unit_cell()`, which uses a different implementation
    (transformations from the PDB file directly).
    """
    pdb_file = pdb.PDBFile.read(pdb_path)
    if pdb_file.get_remark(290) is None:
        # File does not contain a crystal structure -> skip
        return
    ref_unit_cell = pdb_file.get_unit_cell(model=1)

    pdbx_file = pdbx.BinaryCIFFile.read(pdb_path.with_suffix(".bcif"))
    # The reference transformations do not center the copies,
    # so we do not do this here either
    test_unit_cell = pdbx.get_unit_cell(pdbx_file, model=1, center=False)

    for category in ref_unit_cell.get_annotation_categories():
        assert (
            test_unit_cell.get_annotation(category).tolist()
            == ref_unit_cell.get_annotation(category).tolist()
        )
    # The copies are not necessarily in the same order
    # -> Compare each asymmetric unit in the unit cell separately and expect to have
    #    one match
    sym_id_masks = [
        test_unit_cell.sym_id == sym_id for sym_id in np.unique(test_unit_cell.sym_id)
    ]
    distance_matrix = np.full((len(sym_id_masks), len(sym_id_masks)), np.nan)
    for i, mask_i in enumerate(sym_id_masks):
        for j, mask_j in enumerate(sym_id_masks):
            distance_matrix[i, j] = np.mean(
                struc.distance(test_unit_cell[mask_i], ref_unit_cell[mask_j])
            )
    # Expect one match for each asymmetric unit
    assert np.all(np.any(distance_matrix < 1e-3, axis=0))


@pytest.mark.parametrize(
    "path, use_ideal_coord",
    itertools.product(
        glob.glob(join(data_dir("structure"), "molecules", "*.cif")), [False, True]
    ),
)
def test_component_conversion(tmpdir, path, use_ideal_coord):
    """
    After reading a component from a CIF file, writing the component
    back to a new file and reading it again should give the same
    structure.
    """
    cif_file = pdbx.CIFFile.read(path)
    ref_atoms = pdbx.get_component(cif_file, use_ideal_coord=use_ideal_coord)

    cif_file = pdbx.CIFFile()
    pdbx.set_component(cif_file, ref_atoms, data_block="test")
    file_path = join(tmpdir, "test")
    cif_file.write(file_path)

    cif_file = pdbx.CIFFile.read(path)
    test_atoms = pdbx.get_component(cif_file, use_ideal_coord=use_ideal_coord)

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
    sequences_1 = pdbx.get_sequence(pdbx_file)
    pdbx_file = File.read(join(data_dir("structure"), f"4gxy.{format}"))
    sequences_2 = pdbx.get_sequence(pdbx_file)
    assert str(sequences_1["T"]) == "CCGACGGCGCATCAGC"
    assert type(sequences_1["T"]) is seq.NucleotideSequence
    assert str(sequences_1["P"]) == "GCTGATGCGCC"
    assert type(sequences_1["P"]) is seq.NucleotideSequence
    assert str(sequences_1["D"]) == "GTCGG"
    assert type(sequences_1["D"]) is seq.NucleotideSequence
    assert (
        str(sequences_1["A"]) == "MSKRKAPQETLNGGITDMLTELANFEKNVSQAIHKYN"
        "AYRKAASVIAKYPHKIKSGAEAKKLPGVGTKIAEKIDEFLATGKLRKLEKIRQD"
        "DTSSSINFLTRVSGIGPSAARKFVDEGIKTLEDLRKNEDKLNHHQRIGLKYFGD"
        "FEKRIPREEMLQMQDIVLNEVKKVDSEYIATVCGSFRRGAESSGDMDVLLTHPS"
        "FTSESTKQPKLLHQVVEQLQKVHFITDTLSKGETKFMGVCQLPSKNDEKEYPHR"
        "RIDIRLIPKDQYYCGVLYFTGSDIFNKNMRAHALEKGFTINEYTIRPLGVTGVA"
        "GEPLPVDSEKDIFDYIQWKYREPKDRSE"
    )
    assert type(sequences_1["A"]) is seq.ProteinSequence
    assert (
        str(sequences_2["A"]) == "GGCGGCAGGTGCTCCCGACCCTGCGGTCGGGAGTTAA"
        "AAGGGAAGCCGGTGCAAGTCCGGCACGGTCCCGCCACTGTGACGGGGAGTCGCC"
        "CCTCGGGATGTGCCACTGGCCCGAAGGCCGGGAAGGCGGAGGGGCGGCGAGGAT"
        "CCGGAGTCAGGAAACCTGCCTGCCGTC"
    )
    assert type(sequences_2["A"]) is seq.NucleotideSequence


def test_get_sse():
    """
    Check if the secondary structure elements are returned for a short structure
    where the reference secondary structure elements are taken from Mol*
    (https://www.rcsb.org/3d-view/1AKI).
    """
    TOTAL_LENGTH = 129
    # Ranges are given in terms of residue IDs, and the end is inclusive
    HELIX_RANGES = [
        (5, 14),
        (20, 22),
        (25, 36),
        (80, 84),
        (89, 101),
        (104, 107),
        (109, 114),
        (120, 123),
    ]
    SHEET_RANGES = [
        (43, 45),
        (51, 53),
    ]

    ref_sse = np.full(TOTAL_LENGTH, "c", dtype="U1")
    for helix_range in HELIX_RANGES:
        # Conver to zero-based indexing
        ref_sse[helix_range[0] - 1 : helix_range[1]] = "a"
    for sheet_range in SHEET_RANGES:
        ref_sse[sheet_range[0] - 1 : sheet_range[1]] = "b"

    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), "1aki.bcif"))
    test_sse = pdbx.get_sse(pdbx_file)["A"]

    assert test_sse.tolist() == ref_sse.tolist()


@pytest.mark.parametrize("path", glob.glob(join(data_dir("structure"), "*.bcif")))
def test_get_sse_length(path):
    """
    If `match_model` is set in :func:`get_sse()`, the length of the returned array
    must match the number of residues in the structure.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(path)
    atoms = pdbx.get_structure(pdbx_file, model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    sse = pdbx.get_sse(pdbx_file, match_model=1)

    for chain_id in np.unique(atoms.chain_id):
        chain = atoms[atoms.chain_id == chain_id]
        assert len(sse[chain_id]) == struc.get_residue_count(chain)


def test_bcif_encoding():
    """
    Check if encoding and subsequent decoding data in a BinaryCIF file
    restores the original data.
    """
    PDB_ID = "1aki"

    encodings_used = {
        encoding: False
        for encoding in [
            pdbx.ByteArrayEncoding,
            pdbx.FixedPointEncoding,
            # This encoding is not used in the test file
            # pdbx.IntervalQuantizationEncoding,
            pdbx.RunLengthEncoding,
            pdbx.DeltaEncoding,
            pdbx.IntegerPackingEncoding,
            pdbx.StringArrayEncoding,
        ]
    }

    bcif_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), f"{PDB_ID}.bcif"))
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
            except Exception:
                raise Exception(f"Encoding failed for '{category_name}.{column_name}'")

    # Check if each encoding was used at least once
    # to ensure that the test was thorough
    for key, was_used in encodings_used.items():
        try:
            assert was_used
        except Exception:
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
                    assert (
                        cif_column.mask.array.tolist()
                        == bcif_column.mask.array.tolist()
                    )
                # In CIF format, all vales are strings
                # -> ensure consistency
                dtype = bcif_column.data.array.dtype
                assert cif_column.as_array(dtype).tolist() == pytest.approx(
                    bcif_column.as_array(dtype).tolist()
                )
            except Exception:
                raise Exception(
                    f"Comparison failed for '{category_name}.{column_name}'"
                )


@pytest.mark.parametrize(
    "format, create_new_encoding",
    [
        ("cif", None),
        ("bcif", False),
        ("bcif", True),
    ],
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
            test_category = pdbx.CIFCategory.deserialize(ref_category.serialize())
        elif format == "bcif":
            # Access each column to force otherwise lazy deserialization
            for _ in ref_category.values():
                pass
            if create_new_encoding:
                ref_category = _clear_encoding(ref_category)
            test_category = pdbx.BinaryCIFCategory.deserialize(ref_category.serialize())
        try:
            for key in test_category.keys():
                assert ref_category[key] == test_category[key]
        except Exception:
            raise Exception(f"Comparison failed for '{category_name}.{key}'")


@pytest.mark.parametrize(
    "format, level", itertools.product(["cif", "bcif"], ["block", "category", "column"])
)
def test_editing(tmpdir, format, level):
    """
    Check if editing an existing PDBx file works, by checking if replacing some
    category/block/column with a copy of itself does not affect the content.
    """
    File = pdbx.CIFFile if format == "cif" else pdbx.BinaryCIFFile
    Block = File.subcomponent_class()
    Category = Block.subcomponent_class()
    Column = Category.subcomponent_class()

    column = Column(["a", "b", "c"])
    category = Category({"foo_col": column, "bar_col": column, "baz_col": column})
    block = Block({"foo_cat": category, "bar_cat": category, "baz_cat": category})
    ref_pdbx_file = File({"foo_block": block, "bar_block": block, "baz_block": block})
    ref_pdbx_file.write(join(tmpdir, f"original.{format}"))

    pdbx_file = File.read(join(tmpdir, f"original.{format}"))
    if level == "block":
        # Replace block in the mid,
        # to check if the block before and after remain the same
        pdbx_file["bar_block"] = pdbx_file["bar_block"]
    elif level == "category":
        pdbx_file["bar_block"]["bar_cat"] = pdbx_file["bar_block"]["bar_cat"]
    elif level == "column":
        pdbx_file["bar_block"]["bar_cat"]["bar_col"] = pdbx_file["bar_block"][
            "bar_cat"
        ]["bar_col"]
    pdbx_file.write(join(tmpdir, f"edited.{format}"))

    test_pdbx_file = File.read(join(tmpdir, f"edited.{format}"))
    # As the content should not have changed, the serialized files should be identical
    assert test_pdbx_file.serialize() == ref_pdbx_file.serialize()


def test_compress_data():
    """
    Check if the size of :class:`BinaryCIFData` compressed with :func:`compress()` is
    at least as small as the original data compressed by the RCSB PDB.
    """
    PDB_ID = "1aki"

    path = join(data_dir("structure"), f"{PDB_ID}.bcif")
    bcif_file = pdbx.BinaryCIFFile.read(path)
    for category_name, category in bcif_file.block.items():
        for column_name, column in category.items():
            try:
                for attr_name, data in [("data", column.data), ("mask", column.mask)]:
                    if data is None:
                        continue
                    ref_size = len(
                        msgpack.packb(
                            data.serialize(), use_bin_type=True, default=encode_numpy
                        )
                    )
                    # Remove original encoding
                    # and find a new encoding with good compression
                    compressed_data = pdbx.compress(pdbx.BinaryCIFData(data.array))
                    serialized_compressed_data = compressed_data.serialize()
                    # The compressed data should be as small as the original data
                    test_size = len(serialized_compressed_data["data"])
                    test_size = len(
                        msgpack.packb(
                            serialized_compressed_data,
                            use_bin_type=True,
                            default=encode_numpy,
                        )
                    )
                    assert test_size <= ref_size
                    # Check if the data is unaltered after compression
                    restored_data = pdbx.BinaryCIFData.deserialize(
                        serialized_compressed_data
                    )
                    assert restored_data.array.tolist() == data.array.tolist()
            except AssertionError:
                raise AssertionError(f"{category_name}.{column_name} {attr_name}")


@pytest.mark.parametrize(
    "path",
    glob.glob(join(data_dir("structure"), "*.bcif")),
)
def test_compress_file(path):
    """
    Check if the content read from a BinaryCIF file created by :func:`compress()`, is
    the same as from the uncompressed file, while the file size it at least as small
    as the file compressed by the RCSB PDB.
    """
    # Use a relatively high precision to increase strictness of of the equality check
    ATOL = 1e-5
    RTOL = 1e-10

    orig_file = pdbx.BinaryCIFFile.read(path)

    # Create an equivalent file without the original encoding
    uncompressed_file = pdbx.BinaryCIFFile()
    block = pdbx.BinaryCIFBlock()
    for category_name, category in orig_file.block.items():
        block[category_name] = _clear_encoding(category)
    uncompressed_file["block"] = block

    compressed_file = pdbx.compress(uncompressed_file, rtol=RTOL, atol=ATOL)
    # Remove any cached data columns by re-serializing and deserializing
    compressed_file = pdbx.BinaryCIFFile.deserialize(compressed_file.serialize())

    # Check if the data is unaltered after compression
    # Direct equality check is not possible, as the encoding may be different
    for category_name, category in orig_file.block.items():
        for column_name, column in category.items():
            for attr_name, ref_data in [("data", column.data), ("mask", column.mask)]:
                test_data = getattr(
                    compressed_file.block[category_name][column_name], attr_name
                )
                try:
                    if ref_data is None:
                        assert test_data is None
                    else:
                        if np.issubdtype(ref_data.array.dtype, np.floating):
                            # The reference may have used direct ByteArrayEncoding,
                            # which is not able to exactly represent the correct data
                            # due to numerical inaccuracies
                            # -> Expect very small differences
                            # Furthermore, the compression may have different precision
                            assert np.allclose(
                                test_data.array, ref_data.array, rtol=RTOL, atol=ATOL
                            )
                        else:
                            assert test_data.array.tolist() == ref_data.array.tolist()
                except AssertionError:
                    raise AssertionError(f"{category_name}.{column_name} {attr_name}")

    # Check if the file size is at least as small as the original file
    assert _file_size(compressed_file) <= _file_size(orig_file)


@pytest.mark.parametrize("value", [1e10, 1e-10, np.nan, np.inf, -np.inf])
def test_extreme_float_compression(value):
    """
    Check if :func:`compress()` correctly falls back to direct byte encoding of floats
    in extreme cases where fixed point encoding would lead to integer
    underflow/overflow or the value could not be represented by an integer.
    """
    # Not only very small/large values, but a large difference between the values are
    # required, to make fixed point encoding fail
    ref_array = np.array([value, 1.0])

    compressed_data = pdbx.compress(pdbx.BinaryCIFData(ref_array), atol=0)
    serialized_compressed_data = compressed_data.serialize()
    data = pdbx.BinaryCIFData.deserialize(serialized_compressed_data)

    # Check that no fixed point encoding was used
    assert len(data.encoding) == 1
    assert type(data.encoding[0]) is pdbx.ByteArrayEncoding
    assert data.array.tolist() == pytest.approx(ref_array.tolist(), nan_ok=True)


@pytest.mark.parametrize(
    "number, ref_decimals",
    [
        (1.0, 0),
        (1.23, 2),
        (0.001, 3),
        (0.0012345, 4),
        (12300, -2),
        (123.45, 2),
        (123.45678, 4),
        (123.00001, 0),
        (0.00001, 0),
        (0.0, 0),
    ],
)
def test_decimal_places(number, ref_decimals):
    """
    Check if :func`:_get_decimal_places()` returns the correct number of decimal places
    for known examples.
    """
    test_decimals = get_decimal_places(np.array([number]), 1e-6, 1e-4)
    assert test_decimals == ref_decimals


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


def _file_size(bcif_file):
    written_file = BytesIO()
    bcif_file.write(written_file)
    written_file.seek(0)
    return len(written_file.read())


def test_writing_and_reading_extra_fields(tmpdir):
    """
    Check if writing and reading extra fields works.
    """
    # Set up a custom atom array with an additional annotation
    cif_file = pdbx.CIFFile.read(join(data_dir("structure"), "5ugo.cif"))
    atoms = pdbx.get_structure(cif_file)
    custom_annotation = np.arange(atoms.array_length())
    atoms.set_annotation("my_custom_annotation", custom_annotation)

    # Write to file
    new_cif_file = pdbx.CIFFile()
    pdbx.set_structure(new_cif_file, atoms, extra_fields=["my_custom_annotation"])
    new_cif_file.write(join(tmpdir, "test.cif"))

    # Read again
    atoms = pdbx.get_structure(
        pdbx.CIFFile.read(join(tmpdir, "test.cif")),
        extra_fields=["my_custom_annotation"],
    )
    assert "my_custom_annotation" in atoms.get_annotation_categories()
    assert np.all(
        atoms.get_annotation("my_custom_annotation").astype(int) == custom_annotation
    )
