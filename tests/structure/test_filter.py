# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
from os.path import join
from ..util import data_dir
import pytest

@pytest.fixture
def sample_protein():
    return strucio.load_structure(
        join(data_dir("structure"), "3o5r.mmtf"),
        extra_fields = ["atom_id"]
    )

@pytest.fixture
def sample_nucleotide():
    return strucio.load_structure(
        join(data_dir("structure"), "5ugo.mmtf")
    )

@pytest.fixture
def sample_all_atloc_structure():
    return strucio.load_structure(
        join(data_dir("structure"), "1o1z.mmtf"),
        altloc="all"
    )

def test_solvent_filter(sample_protein):
    assert len(sample_protein[struc.filter_solvent(sample_protein)]) == 287

def test_amino_acid_filter(sample_protein):
    assert len(sample_protein[struc.filter_amino_acids(sample_protein)]) == 982

def test_backbone_filter(sample_protein):
    assert len(sample_protein[struc.filter_backbone(sample_protein)]) == 384

def test_intersection_filter(sample_protein):
    assert len(sample_protein[:200][
        struc.filter_intersection(sample_protein[:200],sample_protein[100:])
    ]) == 100

def test_nucleotide_filter(sample_nucleotide):

    assert len(
        sample_nucleotide[struc.filter_nucleotides(sample_nucleotide)]
    ) == 651

def test_filter_first_altloc(sample_all_atloc_structure):
    """
    For a correctly altloc filtered structure no atom should be missing
    and no atom should be present twice.
    """
    ref_atom_set = set()
    for atom_tuple in zip(
        sample_all_atloc_structure.chain_id,
        sample_all_atloc_structure.res_id,
        sample_all_atloc_structure.ins_code,
        sample_all_atloc_structure.atom_name
    ):
        ref_atom_set.add(atom_tuple)
    
    filtered_structure = sample_all_atloc_structure[struc.filter_first_altloc(
        sample_all_atloc_structure,
        sample_all_atloc_structure.altloc_id
    )]
    test_atom_set = set()
    for atom_tuple in zip(
        filtered_structure.chain_id,
        filtered_structure.res_id,
        filtered_structure.ins_code,
        filtered_structure.atom_name
    ):
        try:
            # No atom should be present twice
            assert atom_tuple not in test_atom_set
        except AssertionError:
            print(f"Atom {atom_tuple} is present twice")
            raise
        test_atom_set.add(atom_tuple)
    
    # No atom should be missing
    assert test_atom_set == ref_atom_set