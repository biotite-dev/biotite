# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite
import itertools
import numpy as np
import glob
from os.path import join
from .util import data_dir
import pytest
from pytest import approx


@pytest.mark.parametrize(
    "category, key, exp_value", 
    [
        (
            "audit_author", "name",
            ["Neidigh, J.W.", "Fesinmeyer, R.M.", "Andersen, N.H."]
        ),
        (
            "struct_ref_seq", "pdbx_PDB_id_code", "1L2Y"
        ),
        (
            "pdbx_nmr_ensemble", "conformer_selection_criteria",
            "structures with acceptable covalent geometry, "
            "structures with the least restraint violations"
        )
    ]
)
def test_parsing(category, key, exp_value):
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(join(data_dir, "1l2y.cif"))
    cat_dict = pdbx_file[category]
    value = cat_dict[key]
    if isinstance(value, np.ndarray):
        assert value.tolist() == exp_value
    else:
        assert value == exp_value


@pytest.mark.parametrize(
    "string, use_array",
    itertools.product(
        ["", " ", "\n", "\t"],
        [False, True]
    )
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
        category_dict={
            "test_field": ref_value
        }
    )
    
    test_value = pdbx_file["test_category"]["test_field"]
    
    if use_array:
        assert test_value.tolist() == ["."] * LENGTH
    else:
        assert test_value == "."


@pytest.mark.parametrize(
    "path, single_model",
    itertools.product(
        glob.glob(join(data_dir, "*.cif")),
        [False, True]
    )
)
def test_conversion(path, single_model):
    model = 1 if single_model else None
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(path)
    array1 = pdbx.get_structure(pdbx_file, model=model)
    pdbx_file = pdbx.PDBxFile()
    pdbx.set_structure(pdbx_file, array1, data_block="test")
    array2 = pdbx.get_structure(pdbx_file, model=model)
    if array1.box is not None:
        assert np.allclose(array1.box, array2.box)
    assert array1.bonds == array2.bonds
    for category in array1.get_annotation_categories():
        assert array1.get_annotation(category).tolist() == \
               array2.get_annotation(category).tolist()
    assert array1.coord.tolist() == array2.coord.tolist()


def test_extra_fields():
    path = join(data_dir, "1l2y.cif")
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(path)
    stack1 = pdbx.get_structure(pdbx_file, extra_fields=["atom_id","b_factor",
                                "occupancy","charge"])
    pdbx_file = pdbx.PDBxFile()
    pdbx.set_structure(pdbx_file, stack1, data_block="test")
    stack2 = pdbx.get_structure(pdbx_file, extra_fields=["atom_id","b_factor",
                                "occupancy","charge"])
    assert stack1 == stack2


    path = join(data_dir, "1l2y.cif")
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(path)
    stack1 = pdbx.get_structure(
        pdbx_file,
        extra_fields=[
            "atom_id", "b_factor", "occupancy", "charge"
        ]
    )

    pdbx_file = pdbx.PDBxFile()
    pdbx.set_structure(pdbx_file, stack1, data_block="test")
    
    stack2 = pdbx.get_structure(
        pdbx_file,
        extra_fields=[
            "atom_id", "b_factor", "occupancy", "charge"
        ]
    )
    
    assert stack1.ins_code.tolist() == stack2.ins_code.tolist()
    assert stack1.atom_id.tolist() == stack2.atom_id.tolist()
    assert stack1.b_factor.tolist() == approx(stack2.b_factor.tolist())
    assert stack1.occupancy.tolist() == approx(stack2.occupancy.tolist())
    assert stack1.charge.tolist() == stack2.charge.tolist()
    assert stack1 == stack2


def test_unequal_lengths():
    valid_category_dict = {
        "foo1" : ["1", "2", "3"],
        "foo2" : ["1", "2", "3"]
    }
    # Arrays have unequal lengths -> invalid
    invalid_category_dict = {
        "foo1" : ["1", "2", "3"],
        "foo2" : ["1", "2", "3", "4"]
    }
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.set_category("test", valid_category_dict,  block="test_block")
    with pytest.raises(ValueError):
        pdbx_file.set_category(
            "test", invalid_category_dict, block="test_block"
        )
            
        
def test_list_assemblies():
    """
    Test the :func:`list_assemblies()` function based on a known
    example.
    """
    path = join(data_dir, "1f2n.cif")
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(path)

    assembly_list = pdbx.get_assembly_list(pdbx_file)
    assert assembly_list == {
        "1": "complete icosahedral assembly",
        "2": "icosahedral asymmetric unit",
        "3": "icosahedral pentamer",
        "4": "icosahedral 23 hexamer",
        "5": "icosahedral asymmetric unit, std point frame",
        "6": "crystal asymmetric unit, crystal frame"
    }


@pytest.mark.parametrize("single_model", [False, True])
def test_get_assembly(single_model):
    """
    Test whether the :func:`get_assembly()` function produces the same
    number of peptide chains as the
    ``_pdbx_struct_assembly.oligomeric_count`` field indicates.
    """
    model = 1 if single_model else None

    path = join(data_dir, "1f2n.cif")
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(path)

    assembly_category = pdbx_file.get_category(
        "pdbx_struct_assembly", expect_looped=True
    )
    # Test each available assembly
    for id, ref_oligomer_count in zip(
        assembly_category["id"],
        assembly_category["oligomeric_count"]
    ):    
        assembly = pdbx.get_assembly(pdbx_file, assembly_id=id, model=model)
        protein_assembly = assembly[..., struc.filter_amino_acids(assembly)]
        test_oligomer_count = struc.get_chain_count(protein_assembly)

        if single_model:
            assert isinstance(assembly, struc.AtomArray)
        else:
            assert isinstance(assembly, struc.AtomArrayStack)
        assert test_oligomer_count == int(ref_oligomer_count)