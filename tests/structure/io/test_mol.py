# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import datetime
import glob
import itertools
from os.path import join, splitext
from tempfile import TemporaryFile
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.mol as mol
import biotite.structure.io.pdbx as pdbx
from biotite.structure.bonds import BondType
from biotite.structure.io.mol.ctab import BOND_TYPE_MAPPING_REV
from tests.util import data_dir


def list_v2000_sdf_files():
    return [
        path
        for path in glob.glob(join(data_dir("structure"), "molecules", "*.sdf"))
        if "v3000" not in path
    ]


def list_v3000_sdf_files():
    return glob.glob(join(data_dir("structure"), "molecules", "*v3000.sdf"))


def list_cif_files():
    return glob.glob(join(data_dir("structure"), "molecules", "*.cif"))


def toy_atom_array(n_atoms):
    atoms = struc.AtomArray(n_atoms)
    atoms.coord[:] = 1.0
    atoms.element[:] = "H"
    atoms.add_annotation("charge", dtype=int)
    atoms.charge[:] = 0
    atoms.bonds = struc.BondList(n_atoms)
    return atoms


def test_header_conversion():
    """
    Write known example data to the header of a file and expect
    to retrieve the same information when reading the file again.
    """
    ref_header = mol.Header(
        mol_name="TestMol",
        initials="JD",
        program="Biotite",
        time=datetime.datetime.now().replace(second=0, microsecond=0),
        dimensions="3D",
        scaling_factors="Lorem",
        energy="Ipsum",
        registry_number="123",
        comments="Lorem ipsum dolor sit amet",
    )

    record = mol.SDRecord(header=ref_header)
    sdf_file = mol.SDFile({ref_header.mol_name: record})
    temp = TemporaryFile("w+")
    sdf_file.write(temp)

    temp.seek(0)
    sdf_file = mol.SDFile.read(temp)
    test_header = sdf_file.record.header
    temp.close()

    assert test_header == ref_header


@pytest.mark.parametrize(
    "FileClass, path, version, omit_charge, use_charge_property",
    itertools.product(
        [mol.MOLFile, mol.SDFile],
        list_v2000_sdf_files(),
        ["V2000", "V3000"],
        [False, True],
        [False, True],
    ),
)
def test_structure_conversion_from_file(
    FileClass,  # noqa: N803
    path,
    version,
    omit_charge,
    use_charge_property,
):
    """
    After reading a file, writing the structure back to a new file
    and reading it again should give the same structure.

    For :class:`MOLFile` also an SDF file is used, as it is compatible.
    """
    mol_file = FileClass.read(path)
    ref_atoms = mol.get_structure(mol_file)
    if omit_charge:
        ref_atoms.del_annotation("charge")

    mol_file = FileClass()
    mol.set_structure(mol_file, ref_atoms, version=version)
    temp = TemporaryFile("w+")
    mol_file.write(temp)

    if version == "V2000":
        if use_charge_property:
            # Enforce usage of 'M  CHG' entries
            _delete_charge_columns(temp)
        else:
            # Enforce usage of charge column in atom block
            _delete_charge_property(temp)

    temp.seek(0)
    mol_file = FileClass.read(temp)
    test_atoms = mol.get_structure(mol_file)
    if omit_charge:
        assert np.all(test_atoms.charge == 0)
        test_atoms.del_annotation("charge")
    temp.close()

    assert test_atoms == ref_atoms


@pytest.mark.parametrize(
    "FileClass, component_name, version, omit_charge, use_charge_property",
    itertools.product(
        [mol.MOLFile, mol.SDFile],
        [
            "ALA",  # Alanine
            "BNZ",  # Benzene (has aromatic bonds)
            "3P8",  # Methylammonium ion (has charge)
            "MCH",  # Trichloromethane (has element with multiple letters)
        ],
        ["V2000", "V3000"],
        [False, True],
        [False, True],
    ),
)
def test_structure_conversion_to_file(
    FileClass,  # noqa: N803
    component_name,
    version,
    omit_charge,
    use_charge_property,
):
    """
    Writing a component to a file and reading it again should give the same
    structure.
    """
    ref_atoms = info.residue(component_name)

    mol_file = FileClass()
    mol.set_structure(mol_file, ref_atoms, version=version)
    temp = TemporaryFile("w+")
    mol_file.write(temp)

    if version == "V2000":
        if use_charge_property:
            # Enforce usage of 'M  CHG' entries
            _delete_charge_columns(temp)
        else:
            # Enforce usage of charge column in atom block
            _delete_charge_property(temp)

    temp.seek(0)
    mol_file = FileClass.read(temp)
    test_atoms = mol.get_structure(mol_file)
    temp.close()

    assert np.all(test_atoms.element == ref_atoms.element)
    assert np.all(test_atoms.charge == ref_atoms.charge)
    assert np.allclose(test_atoms.coord, ref_atoms.coord)
    assert test_atoms.bonds == ref_atoms.bonds


@pytest.mark.parametrize(
    "path",
    [
        file
        for file in list_v2000_sdf_files() + list_v3000_sdf_files()
        if file.split(".")[0] + ".cif" in list_cif_files()
    ],
)
def test_pdbx_consistency(path):
    """
    Check if the structure parsed from a file is equal to the same
    structure read in PDBx format.
    """
    # Remove '.sdf' and optional '.v3000' suffix
    cif_path = splitext(splitext(path)[0])[0] + ".cif"

    pdbx_file = pdbx.CIFFile.read(cif_path)
    ref_atoms = pdbx.get_component(pdbx_file)
    # The PDBx test files contain information about aromatic bond types,
    # but the SDF test files do not
    ref_atoms.bonds.remove_aromaticity()

    sdf_file = mol.SDFile.read(path)
    test_atoms = mol.get_structure(sdf_file)

    assert test_atoms.coord.shape == ref_atoms.coord.shape
    assert test_atoms.coord.flatten().tolist() == ref_atoms.coord.flatten().tolist()
    assert test_atoms.element.tolist() == ref_atoms.element.tolist()
    assert test_atoms.charge.tolist() == ref_atoms.charge.tolist()
    assert set(tuple(bond) for bond in test_atoms.bonds.as_array()) == set(
        tuple(bond) for bond in ref_atoms.bonds.as_array()
    )


@pytest.mark.parametrize(
    "v2000_path, v3000_path",
    zip(sorted(list_v2000_sdf_files()), sorted(list_v3000_sdf_files())),
)
def test_version_consistency(v2000_path, v3000_path):
    """
    Check if the structure parsed from a `V2000` file is equal to the
    same structure read from a `V3000` file.
    """
    v2000_file = mol.SDFile.read(v2000_path)
    v2000_atoms = v2000_file.record.get_structure()

    v3000_file = mol.SDFile.read(v3000_path)
    v3000_atoms = v3000_file.record.get_structure()

    assert v2000_atoms == v3000_atoms


def test_multi_record_files():
    """
    Check if multiple records in a file can be written and read.
    """
    RES_NAMES = ["TYR", "HWB"]

    ref_atom_arrays = [
        mol.SDFile.read(
            join(data_dir("structure"), "molecules", f"{res_name}.sdf")
        ).record.get_structure()
        for res_name in RES_NAMES
    ]

    sdf_records = {}
    for res_name, ref_atoms in zip(RES_NAMES, ref_atom_arrays):
        record = mol.SDRecord()
        record.set_structure(ref_atoms)
        sdf_records[res_name] = record
    sdf_file = mol.SDFile(sdf_records)
    temp = TemporaryFile("w+")
    sdf_file.write(temp)

    temp.seek(0)
    sdf_file = mol.SDFile.read(temp)
    test_atom_arrays = [sdf_file[res_name].get_structure() for res_name in RES_NAMES]

    assert test_atom_arrays == ref_atom_arrays


def test_metadata_parsing():
    """
    Check if metadata is parsed correctly based on a known example.
    """
    sdf_file = mol.SDFile.read(join(data_dir("structure"), "molecules", "13136.sdf"))
    metadata = sdf_file.record.metadata

    assert metadata["PUBCHEM_COMPOUND_CID"] == "13136"
    assert metadata["PUBCHEM_IUPAC_INCHIKEY"] == "FNAQSUUGMSOBHW-UHFFFAOYSA-H"
    assert metadata["PUBCHEM_COORDINATE_TYPE"] == "1\n5\n255"


def test_metadata_conversion():
    """
    Writing metadata and reading it again should give the same data.
    """
    ref_metadata = {"test_1": "value 1", "test_2": "value 2\nvalue 3"}

    record = mol.SDRecord(metadata=ref_metadata)
    sdf_file = mol.SDFile({"Molecule": record})
    temp = TemporaryFile("w+")
    sdf_file.write(temp)

    temp.seek(0)
    sdf_file = mol.SDFile.read(temp)
    test_metadata = {key.name: val for key, val in sdf_file.record.metadata.items()}
    temp.close()

    assert test_metadata == ref_metadata


@pytest.mark.parametrize(
    "key_string, ref_key_attributes",
    [
        # Cases from Dalby1992
        ("> <MELTING.POINT>", (None, "MELTING.POINT", None, None)),
        ("> 55 (MD-08974) <BOILING.POINT> DT12", (12, "BOILING.POINT", 55, "MD-08974")),
        ("> DT12 55", (12, None, 55, None)),
    ],
)
def test_metadata_key_parsing(key_string, ref_key_attributes):
    """
    Check if metadata keys are parsed correctly based on known examples.
    """
    number, name, registry_internal, registry_external = ref_key_attributes
    ref_key = mol.Metadata.Key(
        number=number,
        name=name,
        registry_internal=registry_internal,
        registry_external=registry_external,
    )

    test_key = mol.Metadata.Key.deserialize(key_string)

    assert test_key == ref_key


@pytest.mark.parametrize("path", list_v2000_sdf_files())
def test_structure_bond_type_fallback(path):
    """
    Check if a bond with a type not supported by MOL files will be translated
    thanks to the bond type fallback in `MolFile.set_structure`
    """
    # Extract original list of bonds from an SDF file
    mol_file = mol.MOLFile.read(path)
    ref_atoms = mol.get_structure(mol_file)
    # Update one bond in `ref_atoms` with a quadruple bond type,
    # which is not supported by SDF files and thus translates to
    # the default bond type
    ref_atoms.bonds.add_bond(0, 1, BondType.QUADRUPLE)
    updated_bond = ref_atoms.bonds.as_array()[
        np.all(ref_atoms.bonds.as_array()[:, [0, 1]] == [0, 1], axis=1)
    ]
    assert updated_bond.tolist()[0][2] == BondType.QUADRUPLE
    test_mol_file = mol.MOLFile()
    mol.set_structure(test_mol_file, ref_atoms)
    # Test bond type fallback to BondType.ANY value (8) in
    # MolFile.set_structure during mol_file.lines formatting
    updated_line = [
        mol_line for mol_line in test_mol_file.lines if mol_line.startswith("  1  2  ")
    ].pop()
    assert int(updated_line[8]) == BOND_TYPE_MAPPING_REV[BondType.ANY]
    # Test bond type fallback to BondType.SINGLE value (1) in
    # MolFile.set_structure during mol_file.lines formatting
    mol.set_structure(test_mol_file, ref_atoms, default_bond_type=BondType.SINGLE)
    updated_line = [
        mol_line for mol_line in test_mol_file.lines if mol_line.startswith("  1  2  ")
    ].pop()
    assert int(updated_line[8]) == BOND_TYPE_MAPPING_REV[BondType.SINGLE]


@pytest.mark.parametrize("atom_type", ["", " ", "A ", " A"])
def test_quoted_atom_types(atom_type):
    """
    Check if V3000 SDF files can handle atom types (aka elements) with
    empty strings or whitespaces.
    """
    ref_atoms = toy_atom_array(1)
    ref_atoms.element[0] = atom_type
    ref_record = mol.SDRecord()
    ref_record.set_structure(ref_atoms, version="V3000")
    sdf_file = mol.SDFile({"Molecule": ref_record})
    temp = TemporaryFile("w+")
    sdf_file.write(temp)

    temp.seek(0)
    sdf_file = mol.SDFile.read(temp)
    test_atoms = sdf_file.record.get_structure()
    assert test_atoms.element[0] == atom_type
    # Also check if the rest of the structure was parsed correctly
    assert test_atoms == ref_atoms


def test_large_structure():
    """
    Check if SDF files automatically switch to V3000 format if the
    number of atoms exceeds the fixed size columns in the table.
    """
    ref_atoms = toy_atom_array(1000)
    sdf_file = mol.SDFile()
    ref_record = mol.SDRecord()
    # Let the SDF file automatically switch to V3000 format
    ref_record.set_structure(ref_atoms, version=None)
    sdf_file = mol.SDFile({"Molecule": ref_record})
    temp = TemporaryFile("w+")
    sdf_file.write(temp)

    temp.seek(0)
    sdf_file = mol.SDFile.read(temp)
    test_atoms = sdf_file.record.get_structure()
    # Check if file is written in V3000 format
    assert "V3000" in str(sdf_file)
    assert test_atoms == ref_atoms


def _delete_charge_columns(file):
    """
    Reset the charge column from a V2000 file to enforce
    usage of `M  CHG` entries.
    """
    CHARGE_START = 36
    CHARGE_STOP = 39

    file.seek(0)
    lines = file.read().splitlines()
    for i, line in enumerate(lines):
        if (
            len(line) >= CHARGE_STOP
            and line[CHARGE_START:CHARGE_STOP].strip()
            and "V2000" not in line
            and "M  CHG" not in line
        ):
            # Line contains a charge value -> reset to 0
            line = line[:CHARGE_START] + "  0" + line[CHARGE_STOP:]
        lines[i] = line
    file.seek(0)
    file.truncate()
    file.write("\n".join(lines) + "\n")


def _delete_charge_property(file):
    """
    Delete the charge properties from from a V2000 file to enforce
    usage of charge column in atom block.
    """
    file.seek(0)
    lines = file.read().splitlines()
    lines = [line for line in lines if not line.startswith("M  CHG")]
    file.seek(0)
    file.truncate()
    file.write("\n".join(lines) + "\n")
