from io import StringIO
from tempfile import TemporaryFile
import pytest
import biotite.sequence as seq
import biotite.sequence.io.genbank as gb
import biotite.sequence.io.gff as gff
from tests.util import data_dir


@pytest.mark.parametrize("order", [(0, 1, 2), (0, 2, 1), (2, 1, 0)])
@pytest.mark.parametrize("strand", ["+", "-"])
def test_discontinuous_feature(order, strand):
    """
    Test whether the lines of a discontinuous feature, i.e. lines that share
    the same ID, are merged into one feature, even if another feature lies
    between them.
    For this purpose a small GFF3 file is created with the lines in
    different orders and on both strands, and the resulting locations and
    strands are compared to the expected ones.
    """
    lines = [
        f"chr1\t.\tCDS\t1\t3\t.\t{strand}\t0\tID=cds1",
        f"chr1\t.\tgene\t4\t6\t.\t{strand}\t.\tID=other",
        f"chr1\t.\tCDS\t7\t9\t.\t{strand}\t0\tID=cds1",
    ]
    gff_file = gff.GFFFile.read(
        StringIO("##gff-version 3\n" + "\n".join(lines[i] for i in order) + "\n")
    )
    annotation = gff.get_annotation(gff_file)
    assert len(annotation) == 2
    features = {feature.qual["ID"]: feature for feature in annotation}
    assert features["cds1"].key == "CDS"
    assert {(loc.first, loc.last) for loc in features["cds1"].locs} == {(1, 3), (7, 9)}
    assert {(loc.first, loc.last) for loc in features["other"].locs} == {(4, 6)}
    expected_strand = (
        seq.Location.Strand.FORWARD if strand == "+" else seq.Location.Strand.REVERSE
    )
    assert all(loc.strand == expected_strand for loc in features["cds1"].locs)


def test_independent_features_without_id():
    """
    Test whether lines without an ID are kept as separate features, instead
    of being merged into one feature.
    """
    gff_file = gff.GFFFile.read(
        StringIO(
            "##gff-version 3\n"
            "chr1\t.\tgene\t1\t3\t.\t+\t.\t.\n"
            "chr1\t.\tgene\t4\t6\t.\t+\t.\t.\n"
        )
    )
    annotation = gff.get_annotation(gff_file)
    assert len(annotation) == 2
    assert all(feature.qual == {} for feature in annotation)


@pytest.mark.parametrize(
    "path", ["bt_lysozyme.gff3", "gg_avidin.gff3", "ec_bl21.gff3", "sc_chrom1.gff3"]
)
def test_conversion_lowlevel(path):
    """
    Test whether the low-level GFF3 interface can properly read
    a GenBank file and write a file, without data changing.
    """
    gff_file = gff.GFFFile.read(data_dir("sequence") / path)
    ref_entries = [entry for entry in gff_file]

    gff_file = gff.GFFFile()
    for entry in ref_entries:
        gff_file.append(entry)
    temp = TemporaryFile("w+")
    gff_file.write(temp)

    temp.seek(0)
    gff_file = gff.GFFFile.read(temp)
    temp.close()
    test_entries = [field for field in gff_file]
    assert test_entries == ref_entries


@pytest.mark.parametrize(
    "path", ["bt_lysozyme.gff3", "gg_avidin.gff3", "ec_bl21.gff3", "sc_chrom1.gff3"]
)
def test_conversion_highlevel(path):
    """
    Test whether the high-level GFF3 interface can properly read
    the features from a GFF3 file and write these properties to a file
    without data changing.
    The 'phase' is tested additionally, since it is not part of a
    `Feature` object.
    """
    gff_file = gff.GFFFile.read(data_dir("sequence") / path)
    ref_annot = gff.get_annotation(gff_file)
    ref_phases = [record.phase for record in gff_file if record.type == "CDS"]

    gff_file = gff.GFFFile()
    gff.set_annotation(gff_file, ref_annot)
    temp = TemporaryFile("w+")
    gff_file.write(temp)

    temp.seek(0)
    gff_file = gff.GFFFile.read(temp)
    temp.close()
    test_annot = gff.get_annotation(gff_file)
    test_phases = [record.phase for record in gff_file if record.type == "CDS"]

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
    gb_file = gb.GenBankFile.read(data_dir("sequence") / path)
    ref_annot = gb.get_annotation(gb_file)

    gff_file = gff.GFFFile.read((data_dir("sequence") / path).with_suffix(".gff3"))
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

    def record(seqid):
        return gff.GFFRecord(seqid, "ab", "cd", 1, 2, None, None, None, {"Id": "foo"})

    file.append(record("a"))
    assert file[0] == record("a")
    file.append(record("b"))
    file.insert(1, record("c"))
    file[1] = record("d")
    file.insert(3, record("e"))
    del file[2]
    assert [record.seqid for record in file] == ["a", "d", "e"]


def test_entry_indexing():
    """
    Test whether a GFF3 file is indexed correctly based on an artificial
    test file with multiple directives, including '##FASTA'.
    """
    with pytest.warns(UserWarning):
        file = gff.GFFFile.read(data_dir("sequence") / "indexing_test.gff3")
    assert file._directives == [
        ("directive 1", 1),
        ("directive 2", 2),
        ("directive 3", 7),
        ("FASTA", 8),
    ]
    assert file._entries == [3, 4, 6]


def test_percent_encoding():
    """
    Test whether percent encoding is working correctly based on an
    artificial test file.
    """
    file = gff.GFFFile.read(data_dir("sequence") / "percent_test.gff3")
    record = file[0]
    assert record.seqid == "123,456"
    assert record.source == "ääh"
    assert record.type == "regi&n"
    assert record.attributes == {
        "ID": "AnID;AnotherID",
        "Name": "Ångström",
        "c$l$r": "red\tgreen\tblue",
    }

    file2 = gff.GFFFile()
    file2.append(record)
    assert file2[0] == record


def test_error():
    """
    Assert that certain exceptions are raised
    """
    file = gff.GFFFile()
    with pytest.raises(ValueError):
        # 'seqid' beginning with '>' is not legal
        file.append(gff.GFFRecord(">xyz", "ab", "cd", 1, 2))
    with pytest.raises(ValueError):
        # String fields must not be empty
        file.append(gff.GFFRecord("", "ab", "cd", 1, 2))
    with pytest.raises(ValueError):
        # String fields must not be empty
        file.append(gff.GFFRecord("xyz", "", "cd", 1, 2))
    with pytest.raises(ValueError):
        # String fields must not be empty
        file.append(gff.GFFRecord("xyz", "ab", "", 1, 2))
    with pytest.raises(TypeError):
        # Only records are accepted
        file.append(("xyz", "ab", "cd", 1, 2, None, None, None, {}))


def test_feature_without_id():
    """
    A feature without 'ID' should raise an error if it has multiple
    locations and consequently multiple entries in the GFF3 file.
    """
    annot = seq.Annotation(
        [
            seq.Feature(
                key="CDS",
                locs=[seq.Location(1, 2), seq.Location(4, 5)],
                qual={"some": "qualifiers"},
            )
        ]
    )
    file = gff.GFFFile()
    with pytest.raises(ValueError):
        gff.set_annotation(file, annot)
