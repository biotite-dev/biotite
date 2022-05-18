# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from tempfile import TemporaryFile
from os.path import join
import biotite.sequence as seq
import biotite.sequence.io.gff as gff
import biotite.sequence.io.genbank as gb
import numpy as np
import pytest
from ..util import data_dir


@pytest.mark.parametrize(
    "path",
    ["bt_lysozyme.gff3", "gg_avidin.gff3", "ec_bl21.gff3", "sc_chrom1.gff3"]
)
def test_conversion_lowlevel(path):
    """
    Test whether the low-level GFF3 interface can properly read
    a GenBank file and write a file, without data changing.
    """
    gff_file = gff.GFFFile.read(join(data_dir("sequence"), path))
    ref_entries = [entry for entry in gff_file]

    gff_file = gff.GFFFile()
    for entry in ref_entries:
        gff_file.append(*entry)
    temp = TemporaryFile("w+")
    gff_file.write(temp)

    temp.seek(0)
    gff_file = gff.GFFFile.read(temp)
    temp.close()
    test_entries = [field for field in gff_file]
    assert test_entries == ref_entries


@pytest.mark.parametrize(
    "path",
    ["bt_lysozyme.gff3", "gg_avidin.gff3", "ec_bl21.gff3", "sc_chrom1.gff3"]
)
def test_conversion_highlevel(path):
    """
    Test whether the high-level GFF3 interface can properly read
    the features from a GFF3 file and write these properties to a file
    without data changing.
    The 'phase' is tested additionally, since it is not part of a
    `Feature` object.
    """
    gff_file = gff.GFFFile.read(join(data_dir("sequence"), path))
    ref_annot = gff.get_annotation(gff_file)
    ref_phases = []
    for _, _, type, _, _, _, _, phase, _ in gff_file:
        if type == "CDS":
            ref_phases.append(phase)

    gff_file = gff.GFFFile()
    gff.set_annotation(gff_file, ref_annot)
    temp = TemporaryFile("w+")
    gff_file.write(temp)

    temp.seek(0)
    gff_file = gff.GFFFile.read(temp)
    temp.close()
    test_annot = gff.get_annotation(gff_file)
    test_phases = []
    for _, _, type, _, _, _, _, phase, _ in gff_file:
        if type == "CDS":
            test_phases.append(phase)
    
    assert ref_annot == test_annot
    assert test_phases == ref_phases


@pytest.mark.parametrize(
    "path", ["bt_lysozyme.gp", "gg_avidin.gb", "ec_bl21.gb", "sc_chrom1.gb"]
)
def test_genbank_consistency(path):
    """
    Test whether the same annotation (if reasonable) can be read from a
    GFF3 file and a GenBank file.
    """
    gb_file = gb.GenBankFile.read(join(data_dir("sequence"), path))
    ref_annot = gb.get_annotation(gb_file)

    gff_file = gff.GFFFile.read(join(data_dir("sequence"), path[:-3] + ".gff3"))
    test_annot = gff.get_annotation(gff_file)
    
    # Remove qualifiers, since they will be different
    # in GFF3 and GenBank
    ref_annot = seq.Annotation(
        [seq.Feature(feature.key, feature.locs) for feature in ref_annot]
    )
    test_annot = seq.Annotation(
        [seq.Feature(feature.key, feature.locs) for feature in test_annot]
    )
    for feature in test_annot:
        # Only CDS, gene, intron and exon should be equal
        # in GenBank and GFF3
        if feature.key in ["CDS", "gene", "intron", "exon"]:
            try:
                assert feature in test_annot
            except AssertionError:
                print(feature.key)
                for loc in feature.locs:
                    print(loc)
                raise


def test_file_access():
    """
    Test getting, setting, deleting and inserting entries in a GFF3
    file.
    """
    file = gff.GFFFile()
    entry_scaffold = ("ab", "cd", 1, 2, None, None, None, {"Id":"foo"})
    entry = ("a",) + entry_scaffold
    file.append(*entry)
    assert file[0] == entry
    file.append(*(("b",) + entry_scaffold))
    file.insert(1, *(("c",) + entry_scaffold))
    file[1] = ("d",) + entry_scaffold
    file.insert(3, *(("e",) + entry_scaffold))
    del file[2]
    assert [seqid for seqid, _, _, _, _, _, _, _, _ in file] \
        == ["a", "d", "e", ]


def test_entry_indexing():
    """
    Test whether a GFF3 file is indexed correctly based on an artificial
    test file with multiple directives, including '##FASTA'.
    """
    with pytest.warns(UserWarning):
        file = gff.GFFFile.read(
            join(data_dir("sequence"), "indexing_test.gff3")
        )
    assert file._directives == [
        ("directive 1", 1),
        ("directive 2", 2),
        ("directive 3", 7),
        ("FASTA", 8),
    ]
    assert file._entries == [3,4,6]



def test_percent_encoding():
    """
    Test whether percent encoding is working correctly based on an
    artificial test file.
    """
    file = gff.GFFFile.read(join(data_dir("sequence"), "percent_test.gff3"))
    seqid, source, type, start, end, score, strand, phase, attrib \
        = file[0]
    assert seqid == "123,456"
    assert source == "ääh"
    assert type == "regi&n"
    assert attrib == {
        "ID"   : "AnID;AnotherID",
        "Name" : "Ångström",
        "c$l$r": "red\tgreen\tblue"
    }

    file2 = gff.GFFFile()
    file.append(seqid, source, type, start, end, score, strand, phase, attrib)
    assert (seqid, source, type, start, end, score, strand, phase, attrib) \
        == file[0]


def test_error():
    """
    Assert that certain exceptions are raised
    """
    file = gff.GFFFile()
    with pytest.raises(ValueError):
        # 'seqid' beginning with '>' is not legal
        file.append(">xyz", "ab", "cd", 1, 2, None, None, None, {"Id":"foo"})
    with pytest.raises(ValueError):
        # String fields must not be empty
        file.append("", "ab", "cd", 1, 2, None, None, None, {"Id":"foo"})
    with pytest.raises(ValueError):
        # String fields must not be empty
        file.append("xyz", "", "cd", 1, 2, None, None, None, {"Id":"foo"})
    with pytest.raises(ValueError):
        # String fields must not be empty
        file.append("xyz", "ab", "", 1, 2, None, None, None, {"Id":"foo"})

def test_feature_without_id():
    """
    A feature without 'ID' should raise an error if it has multiple
    locations and consequently multiple entries in the GFF3 file.
    """
    annot = seq.Annotation(
        [seq.Feature(
            key  = "CDS",
            locs = [seq.Location(1,2), seq.Location(4,5)],
            qual = {"some" : "qualifiers"}
        )]
    )
    file = gff.GFFFile()
    with pytest.raises(ValueError):
        gff.set_annotation(file, annot)