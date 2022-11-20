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
def canonical_sample_protein():
    return strucio.load_structure(
        join(data_dir("structure"), "3o5r.mmtf")
    )

@pytest.fixture
def sample_protein():
    return strucio.load_structure(
        join(data_dir("structure"), "5eil.mmtf")
    )

@pytest.fixture
def canonical_sample_nucleotide():
    return strucio.load_structure(
        join(data_dir("structure"), "5ugo.mmtf")
    )

@pytest.fixture
def sample_nucleotide():
    return strucio.load_structure(
        join(data_dir("structure"), "4p5j.mmtf")
    )

@pytest.fixture
def sample_carbohydrate():
    return strucio.load_structure(
        join(data_dir("structure"), "2d0f.mmtf")
    )

@pytest.fixture
def all_atloc_structure():
    return strucio.load_structure(
        join(data_dir("structure"), "1o1z.mmtf"),
        extra_fields = ["occupancy"],
        altloc="all"
    )

def test_solvent_filter(canonical_sample_protein):
    assert len(canonical_sample_protein[struc.filter_solvent(canonical_sample_protein)]) == 287

def test_canonical_amino_acid_filter(canonical_sample_protein):
    assert (
        len(canonical_sample_protein[
            struc.filter_canonical_amino_acids(canonical_sample_protein)
        ]) == 982
    )

def test_amino_acid_filter(sample_protein):
    assert (
        struc.get_residue_count((sample_protein[
            struc.filter_amino_acids(sample_protein)
        ])) ==
        struc.get_residue_count((sample_protein[
            struc.filter_canonical_amino_acids(sample_protein)
        ])) + 3
    )

def test_canonical_nucleotide_filter(canonical_sample_nucleotide):
    assert (
        len(canonical_sample_nucleotide[
            struc.filter_canonical_nucleotides(canonical_sample_nucleotide)
        ]) == 651
    )

def test_nucleotide_filter(sample_nucleotide):
    assert (
        struc.get_residue_count((sample_nucleotide[
            struc.filter_nucleotides(sample_nucleotide)
        ])) ==
        struc.get_residue_count((sample_nucleotide[
            struc.filter_canonical_nucleotides(sample_nucleotide)
        ])) + 1
    )

def test_carbohydrate_filter(sample_carbohydrate):
    assert (
        struc.get_residue_count((sample_carbohydrate[
            struc.filter_carbohydrates(sample_carbohydrate)
        ])) == 8
    )


def test_peptide_backbone_filter(canonical_sample_protein):
    assert (
        len(canonical_sample_protein[
            struc.filter_peptide_backbone(canonical_sample_protein)
        ]) == 384
    )


def test_phosphate_backbone_filter(canonical_sample_nucleotide):
    # take a chain D with five canonical nucleotides
    # => there should be 5 x 6 = 30 backbone atoms
    chain_d = canonical_sample_nucleotide[
        canonical_sample_nucleotide.chain_id == 'D'
    ]
    assert len(chain_d[struc.filter_phosphate_backbone(chain_d)]) == 30


def test_linear_bond_continuity_filter(canonical_sample_protein):
    # Since there are no discontinuities within the sample protein's backbone
    # the output shouldn't differ from `filter_peptide_backbone` and evaluates
    # to the number of backbone atoms in the protein chain.
    a = canonical_sample_protein
    a_bb = a[struc.filter_peptide_backbone(a)]
    assert len(a_bb[struc.filter_linear_bond_continuity(a_bb)]) == len(a_bb)

    # For the Gly residue, the order of atoms is N, CA, C, O
    # => the molecule is linear, so all four atoms shall be positive.
    gly = a[a.res_id == 13]
    assert len(gly[struc.filter_linear_bond_continuity(gly)]) == 4

    # For an Ala residue, the order is N, CA, C, O, CB
    # Hence, the last atom breaks the continuity
    ala = a[a.res_id == 14]
    assert len(ala[struc.filter_linear_bond_continuity(ala)]) == 4

    # For a Pro residue, there are two groups of consecutive atoms:
    # (1) N, CA, C
    # (2) CB, CG, CD
    # and the O between them breaks the continuity => 6 atoms in total
    pro = a[a.res_id == 15]
    assert len(pro[struc.filter_linear_bond_continuity(pro)]) == 6


def test_polymer_filter(canonical_sample_nucleotide, sample_carbohydrate):
    a = canonical_sample_nucleotide

    # Check for nucleotide filtering
    a_nuc = a[struc.filter_polymer(a, pol_type='n')]
    # Take three nucleic acids chains and remove solvent => the result should
    # encompass all nucleotide polymer atoms, which is exactly the output of the
    # `filter_polymer()`. In the structure file, the filtered atoms are 1-651.
    a_nuc_manual = a[np.isin(a.chain_id, ['D', 'P', 'T']) & ~struc.filter_solvent(a)]
    assert len(a_nuc) == len(a_nuc_manual) == 651
    assert set(a_nuc.chain_id) == {'D', 'P', 'T'}
    # chain D should be absent
    a_nuc = a_nuc[struc.filter_polymer(a_nuc, min_size=6, pol_type='n')]
    assert set(a_nuc.chain_id) == {'P', 'T'}

    # Single protein chain A: residues 10-335
    a_pep = a[struc.filter_polymer(a, pol_type='p')]
    assert len(a_pep) == len(a[(a.res_id >= 10) & (a.res_id <= 335) & (a.chain_id == 'A')])

    # Chain B has five carbohydrate residues
    # Chain C has four
    # => Only chain B is selected
    a = sample_carbohydrate
    a_carb = a[struc.filter_polymer(a, min_size=4, pol_type='carb')]
    assert set(a_carb.chain_id) == {'B'}
    assert struc.get_residue_count(a_carb) == 5


def test_intersection_filter(canonical_sample_protein):
    assert (
        len(canonical_sample_protein[:200][
            struc.filter_intersection(
                canonical_sample_protein[:200],canonical_sample_protein[100:]
            )
        ]) == 100
    )

@pytest.mark.parametrize("filter_func", ["first", "occupancy"])
def test_filter_altloc(all_atloc_structure, filter_func):
    """
    For a correctly altloc filtered structure no atom should be missing
    and no atom should be present twice.
    """
    ref_atom_set = set()
    for atom_tuple in zip(
        all_atloc_structure.chain_id,
        all_atloc_structure.res_id,
        all_atloc_structure.ins_code,
        all_atloc_structure.atom_name
    ):
        ref_atom_set.add(atom_tuple)
    
    if filter_func == "first":
        filtered_structure = all_atloc_structure[struc.filter_first_altloc(
            all_atloc_structure,
            all_atloc_structure.altloc_id
        )]
    elif filter_func == "occupancy":
        filtered_structure = all_atloc_structure[
            struc.filter_highest_occupancy_altloc(
                all_atloc_structure,
                all_atloc_structure.altloc_id,
                all_atloc_structure.occupancy
            )
        ]

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


def test_filter_highest_occupancy_altloc(all_atloc_structure):
    """
    When filtering altlocs with the highest occupancy, the average
    occupancy should be higher than the the average occupancy when the
    first altloc ID is always filtered.
    """
    # Set the occupancy of SECOND altloc ID very high
    all_atloc_structure.occupancy[all_atloc_structure.altloc_id == "B"] = 1.0
    
    # filter_first_altloc
    filtered_structure = all_atloc_structure[struc.filter_first_altloc(
        all_atloc_structure,
        all_atloc_structure.altloc_id
    )]
    ref_occupancy_sum = np.average(filtered_structure.occupancy)

    # filter_highest_occupancy_altloc
    filtered_structure = all_atloc_structure[
        struc.filter_highest_occupancy_altloc(
            all_atloc_structure,
            all_atloc_structure.altloc_id,
            all_atloc_structure.occupancy
        )
    ]
    test_occupancy_sum = np.average(filtered_structure.occupancy)
    
    assert test_occupancy_sum > ref_occupancy_sum