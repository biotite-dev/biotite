# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.sequence as seq
import pytest


@pytest.mark.parametrize("table_id",
    [1,2,3,4,5,6,9,10,11,12,13,14,16,21,22,23,24,25,26,27,28,29,30,31])
def test_table_load(table_id):
    table = seq.CodonTable.load(table_id)


def test_table_indexing():
    table = seq.CodonTable.load("Standard")
    assert table["ATG"] == "M"
    for codon in table["Y"]:
        assert codon in ("TAT", "TAC")
    assert table[(0, 0, 0)] == 8
    for codon in table[8]:
        assert codon in ((0, 0, 0), (0, 0, 2))


def test_incomplete_input_dict():
    """
    Ensure that an exception is raised when the input codon dictionary
    misses an entry.
    """
    with pytest.raises(ValueError, match=".* dictionary does not contain .*"):
        codon_dict = seq.CodonTable.default_table().codon_dict()
        del codon_dict["CTG"]
        seq.CodonTable(codon_dict, starts=["ATG"])



def test_codon_dict():
    """
    Assert that the 'codon_dict()' method returns a correct dictionary.
    """
    default_table = seq.CodonTable.default_table()
    ref_codon_dict = default_table.codon_dict()
    test_codon_dict = seq.CodonTable(
        ref_codon_dict, default_table.start_codons()
    ).codon_dict()
    assert ref_codon_dict == test_codon_dict
