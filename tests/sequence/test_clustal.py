# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import io
import os
import os.path
import pytest
import biotite.sequence as seq
import biotite.sequence.io.clustal as clustal
from tests.util import data_dir


def test_access_low_level():
    """Test reading a ClustalW file and accessing sequences by name."""
    path = os.path.join(data_dir("sequence"), "clustal.aln")
    file = clustal.ClustalFile.read(path)
    assert file["seq1"] == "ADTRCGTARDCGTR-DRTCGRAGD"
    assert file["seq2"] == "ADTRCGT---CGTRADRTCGRAGD"
    assert file["seq3"] == "ADTRCGTARDCGTRADR--GRAGD"
    assert list(file.keys()) == ["seq1", "seq2", "seq3"]


def test_access_multi_block():
    """Test reading a ClustalW file with multiple blocks."""
    path = os.path.join(data_dir("sequence"), "clustal_multi.aln")
    file = clustal.ClustalFile.read(path)
    assert file["seq1"] == ("ADTRCGTARDCGTR-DRTCGRAGDADTRCGTARDCGTR-DRTCGRAGD")
    assert file["seq2"] == ("ADTRCGT---CGTRADRTCGRAGDADTRCGT---CGTRADRTCGRAGD")


def test_dict_interface():
    """Test dictionary-like interface of ClustalFile."""
    file = clustal.ClustalFile()
    file["seq1"] = "ADTRCGT"
    file["seq2"] = "ADTRCGT"
    assert len(file) == 2
    assert "seq1" in file
    assert list(file) == ["seq1", "seq2"]

    del file["seq1"]
    assert len(file) == 1
    assert "seq1" not in file


def test_alignment_conversion():
    """Test conversion between ClustalFile and Alignment object."""
    path = os.path.join(data_dir("sequence"), "clustal.aln")
    file = clustal.ClustalFile.read(path)
    alignment = clustal.get_alignment(file)
    assert str(alignment) == (
        "ADTRCGTARDCGTR-DRTCGRAGD\n"
        "ADTRCGT---CGTRADRTCGRAGD\n"
        "ADTRCGTARDCGTRADR--GRAGD"
    )  # fmt: skip


def test_round_trip():
    """Test writing an alignment and reading it back."""
    path = os.path.join(data_dir("sequence"), "clustal.aln")
    file = clustal.ClustalFile.read(path)
    alignment = clustal.get_alignment(file)

    # Write alignment to a new ClustalFile
    file2 = clustal.ClustalFile()
    clustal.set_alignment(file2, alignment, seq_names=["seq1", "seq2", "seq3"])
    alignment2 = clustal.get_alignment(file2)
    assert alignment == alignment2


def test_write_read_round_trip():
    """Test that writing to a file and reading back yields the same
    alignment."""
    path = os.path.join(data_dir("sequence"), "clustal.aln")
    file = clustal.ClustalFile.read(path)
    alignment = clustal.get_alignment(file)

    # Write to string buffer
    buffer = io.StringIO()
    file2 = clustal.ClustalFile()
    clustal.set_alignment(file2, alignment, seq_names=["seq1", "seq2", "seq3"])
    file2.write(buffer)

    # Read back from string buffer
    buffer.seek(0)
    file3 = clustal.ClustalFile.read(buffer)
    alignment3 = clustal.get_alignment(file3)
    assert alignment == alignment3


def test_name_count_mismatch():
    """Test that mismatched name count raises ValueError."""
    path = os.path.join(data_dir("sequence"), "clustal.aln")
    file = clustal.ClustalFile.read(path)
    alignment = clustal.get_alignment(file)

    file2 = clustal.ClustalFile()
    with pytest.raises(ValueError):
        clustal.set_alignment(file2, alignment, seq_names=["seq1", "seq2"])


@pytest.mark.parametrize("seq_type", (None, seq.ProteinSequence))
def test_seq_type(seq_type):
    """Test explicit sequence type parameter."""
    path = os.path.join(data_dir("sequence"), "clustal.aln")
    file = clustal.ClustalFile.read(path)
    alignment = clustal.get_alignment(file, seq_type=seq_type)
    # Should produce a valid alignment regardless of type
    assert len(alignment.sequences) == 3
