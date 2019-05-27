# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
from os.path import join
import biotite
import biotite.sequence as seq
import biotite.sequence.io.genbank as gb
import numpy as np
import pytest
from .util import data_dir


@pytest.mark.parametrize(
    "path",
    glob.glob(join(data_dir, "*.gb")) + \
    glob.glob(join(data_dir, "[!multifile]*.gp"))
)
def test_contiguous_field_pos(path):
    """
    Check whether the internal index of a GenBankFile is contiguous
    """
    gb_file = gb.GenBankFile()
    gb_file.read(path)
    assert gb_file._field_pos[0][0] == 0
    for i in range(1, len(gb_file._field_pos)):
        start, _, _ = gb_file._field_pos[i]
        _, stop, _  = gb_file._field_pos[i-1]
        assert start == stop


@pytest.mark.parametrize(
    "path",
    glob.glob(join(data_dir, "*.gb")) + \
    glob.glob(join(data_dir, "[!multifile]*.gp"))
)
def test_conversion_lowlevel(path):
    """
    Test whether the low-level GenBank interface can properly read
    a GenBank file and write a file, without data changing.
    """
    gb_file = gb.GenBankFile()
    gb_file.read(path)
    ref_parsed_fields = [field for field in gb_file]

    gb_file = gb.GenBankFile()
    for name, content, subfields in ref_parsed_fields:
        gb_file.append(name, content, subfields)
    temp_file_name = biotite.temp_file("gb")
    gb_file.write(temp_file_name)

    gb_file = gb.GenBankFile()
    gb_file.read(temp_file_name)
    test_parsed_fields = [field for field in gb_file]
    assert test_parsed_fields == ref_parsed_fields


@pytest.mark.parametrize(
    "path",
    glob.glob(join(data_dir, "*.gb")) + \
    glob.glob(join(data_dir, "[!multifile]*.gp"))
)
def test_conversion_highlevel(path):
    """
    Test whether the high-level GenBank interface can properly read
    the locus, annotation and sequence from GenBank file and write
    these properties to a file, without data changing.
    """
    suffix = path[-2:]
    gb_file = gb.GenBankFile()
    gb_file.read(path)
    ref_locus = gb.get_locus(gb_file)
    ref_annot_seq = gb.get_annotated_sequence(gb_file, format=suffix)

    gb_file = gb.GenBankFile()
    gb.set_locus(gb_file, *ref_locus)
    gb.set_annotated_sequence(gb_file, ref_annot_seq)
    temp_file_name = biotite.temp_file("gb")
    gb_file.write(temp_file_name)

    gb_file = gb.GenBankFile()
    gb_file.read(temp_file_name)
    test_locus = gb.get_locus(gb_file)
    test_annot_seq = gb.get_annotated_sequence(gb_file, format=suffix)
    assert test_locus == ref_locus
    assert test_annot_seq.sequence       == ref_annot_seq.sequence
    assert test_annot_seq.annotation     == ref_annot_seq.annotation
    assert test_annot_seq.sequence_start == ref_annot_seq.sequence_start


def test_genbank_utility_gb():
    """
    Check whether the high-level utility functions return the expected
    content of a known GenBank file. 
    """
    gb_file = gb.GenBankFile()
    gb_file.read(join(data_dir, "ec_bl21.gb"))
    assert gb.get_locus(gb_file) \
        == ("CP001509", 4558953, "DNA", True, "BCT", "16-FEB-2017")
    assert gb.get_definition(gb_file) \
        == ("Escherichia coli BL21(DE3), complete genome.")
    assert gb.get_version(gb_file) == "CP001509.3"
    assert gb.get_gi(gb_file) == 296142109
    assert gb.get_db_link(gb_file) \
        == {"BioProject" : "PRJNA20713", "BioSample" : "SAMN02603478"}
    annotation = gb.get_annotation(gb_file, include_only=["CDS"])
    feature = seq.Feature(
        "CDS",
        [seq.Location(5681, 6457, seq.Location.Strand.REVERSE)],
        {"gene": "yaaA", "transl_table": "11"}
    )
    in_annotation = False
    for f in annotation:
        if f.key == feature.key and f.locs == feature.locs and \
           all([(key, val in f.qual.items())
                for key, val in feature.qual.items()]):
                    in_annotation = True
    assert in_annotation
    assert len(gb.get_sequence(gb_file, format="gb")) == 4558953


def test_genbank_utility_gp():
    """
    Check whether the high-level utility functions return the expected
    content of a known GenPept file. 
    """
    gp_file = gb.GenBankFile()
    gp_file.read(join(data_dir, "bt_lysozyme.gp"))
    #[print(e) for e in gp_file._field_pos]
    assert gb.get_locus(gp_file) \
        == ("AAC37312", 147, "", False, "MAM", "27-APR-1993")
    assert gb.get_definition(gp_file) == "lysozyme [Bos taurus]."
    assert gb.get_version(gp_file) == "AAC37312.1"
    assert gb.get_gi(gp_file) == 163334
    annotation = gb.get_annotation(gp_file)
    feature = seq.Feature(
        "Site",
        [seq.Location(start, stop) for start, stop in zip(
            [52,55,62,76,78,81,117,120,125],
            [53,55,62,76,78,81,117,120,126]
        )],
        {"note": "lysozyme catalytic cleft [active]", "site_type": "active"}
    )
    in_annotation = False
    for f in annotation:
        if f.key == feature.key and f.locs == feature.locs and \
           all([(key, val in f.qual.items())
                for key, val in feature.qual.items()]):
                    in_annotation = True
    assert in_annotation
    assert len(gb.get_sequence(gp_file, format="gp")) == 147


def test_multi_file():
    multi_file = gb.MultiFile()
    multi_file.read(join(data_dir, "multifile.gp"))
    accessions = [gb.get_accession(f) for f in multi_file]
    assert accessions == ["1L2Y_A", "3O5R_A", "5UGO_A"]