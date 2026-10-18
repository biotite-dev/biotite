import io
from tempfile import TemporaryFile
import numpy as np
import pytest
import biotite
import biotite.sequence as seq
import biotite.sequence.io.fastq as fastq
from tests.util import data_dir


@pytest.mark.parametrize("chars_per_line", [None, 80])
def test_access(chars_per_line):
    path = data_dir("sequence") / "random.fastq"
    file = fastq.FastqFile.read(path, offset=33, chars_per_line=chars_per_line)
    assert len(file) == 20
    assert list(file.keys()) == [f"Read:{i + 1:02d}" for i in range(20)]
    del file["Read:05"]
    assert len(file) == 19
    assert list(file.keys()) == [f"Read:{i + 1:02d}" for i in range(20) if i + 1 != 5]
    for seq_str, scores in file.values():
        assert len(seq_str) == len(scores)
        assert (scores >= 0).all()
    seq_str = "ACTCGGT"
    scores = np.array([10, 12, 20, 11, 0, 80, 42])
    file["test"] = seq_str, scores
    seq_str2, scores2 = file["test"]
    assert seq_str == seq_str2
    assert np.array_equal(scores, scores2)


@pytest.mark.parametrize("chars_per_line", [None, 80])
def test_conversion(chars_per_line):
    path = data_dir("sequence") / "random.fastq"
    fasta_file = fastq.FastqFile.read(path, offset=33, chars_per_line=chars_per_line)
    ref_content = dict(fasta_file.items())

    fasta_file = fastq.FastqFile(offset=33, chars_per_line=chars_per_line)
    for identifier, (sequence, scores) in ref_content.items():
        fasta_file[identifier] = sequence, scores
    temp = TemporaryFile("w+")
    fasta_file.write(temp)

    temp.seek(0)
    fasta_file = fastq.FastqFile.read(temp, offset=33, chars_per_line=chars_per_line)
    content = dict(fasta_file.items())
    temp.close()

    for identifier in ref_content:
        ref_sequence, ref_scores = ref_content[identifier]
        test_sequence, test_scores = content[identifier]
        assert test_sequence == ref_sequence
        assert np.array_equal(test_scores, ref_scores)


@pytest.mark.parametrize(
    ("member_name", "expected_value"),
    [
        ("SANGER", 33),
        ("SOLEXA", 64),
        ("ILLUMINA_1_3", 64),
        ("ILLUMINA_1_5", 64),
        ("ILLUMINA_1_8", 33),
    ],
)
def test_offset_enum(member_name, expected_value):
    offset = fastq.FastqFile.Offset[member_name]
    assert int(offset) == expected_value

    scores = np.array([0, 1, 2])
    enum_file = fastq.FastqFile(offset)
    enum_file["seq"] = "ACG", scores
    actual_sequence, actual_scores = enum_file["seq"]
    assert actual_sequence == "ACG"
    assert np.array_equal(actual_scores, scores)


def test_rna_conversion():
    sequence = seq.NucleotideSequence("ACGT")
    scores = np.array([0, 0, 0, 0])
    fastq_file = fastq.FastqFile(offset=fastq.FastqFile.Offset.SANGER)
    fastq.set_sequence(fastq_file, sequence, scores, "seq1", as_rna=False)
    fastq.set_sequence(fastq_file, sequence, scores, "seq2", as_rna=True)
    assert fastq_file["seq1"][0] == "ACGT"
    assert fastq_file["seq2"][0] == "ACGU"


@pytest.mark.parametrize(
    "file_name",
    sorted(data_dir("sequence").glob("*.fastq")),
    ids=lambda path: path.name,
)
def test_read_iter(file_name):
    ref_dict = dict(
        fastq.FastqFile.read(file_name, offset=fastq.FastqFile.Offset.SANGER).items()
    )

    test_dict = dict(
        fastq.FastqFile.read_iter(file_name, offset=fastq.FastqFile.Offset.SANGER)
    )

    for (test_id, (test_seq, test_sc)), (ref_id, (ref_seq, ref_sc)) in zip(
        test_dict.items(), ref_dict.items()
    ):
        assert test_id == ref_id
        assert test_seq == ref_seq
        assert (test_sc == ref_sc).all()


@pytest.mark.parametrize("offset", [33, 42, fastq.FastqFile.Offset.SOLEXA])
@pytest.mark.parametrize("chars_per_line", [None, 80])
@pytest.mark.parametrize("n_sequences", [1, 10])
def test_write_iter(offset, chars_per_line, n_sequences):
    """
    Test whether :class:`FastqFile.write()` and
    :class:`FastqFile.write_iter()` produce the same output file for
    random sequences and scores.
    """
    LENGTH_RANGE = (50, 150)
    SCORE_RANGE = (10, 60)

    # Generate random sequences and scores
    rng = np.random.default_rng(0)
    sequences = []
    scores = []
    for i in range(n_sequences):
        seq_length = rng.integers(*LENGTH_RANGE)
        code = rng.integers(
            len(seq.NucleotideSequence.alphabet_unambiguous), size=seq_length
        )
        sequence = seq.NucleotideSequence()
        sequence.code = code
        sequences.append(sequence)
        score = rng.integers(*SCORE_RANGE, size=seq_length)
        scores.append(score)

    fastq_file = fastq.FastqFile(offset, chars_per_line)
    for i, (sequence, score) in enumerate(zip(sequences, scores)):
        identifier = f"seq_{i}"
        fastq_file[identifier] = (str(sequence), score)
    ref_file = io.StringIO()
    fastq_file.write(ref_file)

    test_file = io.StringIO()
    fastq.FastqFile.write_iter(
        test_file,
        (
            (f"seq_{i}", (str(sequence), score))
            for i, (sequence, score) in enumerate(zip(sequences, scores))
        ),
        offset,
        chars_per_line,
    )

    assert test_file.getvalue() == ref_file.getvalue()


@pytest.mark.parametrize(
    ["original_identifiers", "deduplicated_identifiers", "has_duplicates"],
    [
        (["A", "B", "C"], ["A", "B", "C"], False),
        (["A", "A", "A"], ["A", "A_1", "A_2"], True),
        (["A", "B", "A"], ["A", "B", "A_1"], True),
        (["A", "A", "A_1"], ["A", "A_1", "A_1_1"], True),
        (["A", "A_1", "A"], ["A", "A_1", "A_2"], True),
        (
            ["A", "A", "A_3", "A", "A", "A_1", "A_1"],
            ["A", "A_1", "A_3", "A_2", "A_4", "A_1_1", "A_1_2"],
            True,
        ),
    ],
)
@pytest.mark.parametrize("allow_duplicates", [False, True])
def test_duplicate_identifiers(
    allow_duplicates, original_identifiers, deduplicated_identifiers, has_duplicates
):
    """
    Check that reading a file with duplicate identifiers either raises an
    exception or appends ``_<number>`` suffixes to the duplicates,
    depending on `allow_duplicates`.
    """
    file = io.StringIO(
        "".join(
            f"@{identifier}\nACGT\n+\n!!!!\n" for identifier in original_identifiers
        )
    )

    if allow_duplicates:
        fastq_file = fastq.FastqFile.read(file, offset=33, allow_duplicates=True)
        assert list(fastq_file.keys()) == deduplicated_identifiers
    elif has_duplicates:
        with pytest.raises(biotite.InvalidFileError, match="Duplicate identifier"):
            fastq.FastqFile.read(file, offset=33)
    else:
        fastq_file = fastq.FastqFile.read(file, offset=33)
        assert list(fastq_file.keys()) == deduplicated_identifiers
