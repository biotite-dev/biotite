# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import gc
from pathlib import Path
import numpy as np
import pytest
import biotite.application_v2.mmseqs.app as app_module
from biotite.application_v2.mmseqs import (
    AlignmentDatabase,
    FoldseekApp,
    MMseqsApp,
    MSAFormatMode,
    SequenceDatabase,
    create_database_from_sequences,
    get_sequences_from_database,
)
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.seqtypes import ProteinSequence


@pytest.fixture
def sequences():
    """
    Two protein sequences that are similar enough to find each other.
    """
    return {
        "first": "NLYIQWLKDGGPSSGRPPPS",
        "second": "NLYIQWLKDGGPSSGRPPPA",
    }


@pytest.mark.parametrize("seed", range(5))
def test_database_value_properties(seed):
    """
    Every generated database suffix is passed verbatim to the command line.
    """
    rng = np.random.default_rng(seed)
    length = int(rng.integers(0, 9))
    suffix = "".join(rng.choice(list("_abcdefghijklmnopqrstuvwxyz"), size=length))
    app = MMseqsApp("true")
    database = SequenceDatabase(app, suffix=suffix)
    fasta_path = database.path / "output.fasta"

    future = app.convert_to_fasta(database, fasta_path=fasta_path)

    assert str(database.name) in future.command
    assert future.result() == fasta_path


def test_conversion_to_temporary_file():
    """
    Omitting a conversion path returns a readable temporary binary file.
    """
    app = MMseqsApp("true")
    sequence_db = SequenceDatabase(app)
    alignment_db = AlignmentDatabase(app, sequence_db, sequence_db)

    futures = [
        app.convert_to_fasta(sequence_db),
        app.convert_alignments(alignment_db),
    ]
    for future in futures:
        output_file = future.result()
        output_path = Path(output_file.name)

        assert output_file.readable()
        assert "b" in output_file.mode
        assert str(output_path) in future.command
        assert output_path.is_file()

        output_file.close()
        assert not output_path.exists()


def test_create_db_accepts_text_file(tmp_path):
    """
    An open text file can be passed directly as a ``createdb`` input.
    """
    fasta_path = tmp_path / "sequences.fasta"
    fasta_path.write_text(">sequence\nBIQTITE\n")
    app = MMseqsApp("true")

    with fasta_path.open() as fasta_file:
        future = app.create_db(fasta_file)
        assert fasta_file.name in future.command
        future.result()


def test_no_output_path_option():
    """
    Database results are persisted with :meth:`Database.move`, so no command
    allows an output path option.
    """
    option_lists = {
        name: value
        for name, value in vars(app_module).items()
        if name.endswith("_OPTIONS")
    }
    assert len(option_lists) > 0
    for options in option_lists.values():
        assert "output_path" not in options


def test_flat_msa_format_is_disallowed():
    """
    A flat-file MSA format cannot be represented by :class:`MSADatabase`.
    """
    app = MMseqsApp("true")
    sequence_db = SequenceDatabase(app)
    alignment_db = AlignmentDatabase(app, sequence_db, sequence_db)

    with pytest.raises(ValueError, match="flat file"):
        app.result_to_msa(alignment_db, msa_format_mode=MSAFormatMode.STOCKHOLM)


def test_incompatible_database():
    """
    A database cannot cross the MMseqs2/Foldseek application boundary.
    """
    database = SequenceDatabase(FoldseekApp("true"))
    with pytest.raises(ValueError, match="not compatible"):
        MMseqsApp("true").convert_to_fasta(database, "output.fasta")


def test_tmp_dir_is_instance_specific(tmp_path):
    """
    Each application instance has its own configurable temporary root directory.
    """
    tmp_dir = tmp_path / "mmseqs"

    mmseqs_app = MMseqsApp("true")
    foldseek_app = FoldseekApp("true")
    mmseqs_app.tmp_dir = tmp_dir

    assert mmseqs_app.tmp_dir == tmp_dir
    assert foldseek_app.tmp_dir != tmp_dir
    assert tmp_dir.is_dir()


@pytest.mark.parametrize(
    ["app_class", "prefix"],
    [(MMseqsApp, "mmseqs_"), (FoldseekApp, "foldseek_")],
)
def test_search_uses_isolated_temporary_work_directories(app_class, prefix, tmp_path):
    """
    Concurrent searches use distinct work directories below the configured root.
    """
    app = app_class("true")
    app.tmp_dir = tmp_path
    database = SequenceDatabase(app)

    futures = [app.search(database, database) for _ in range(2)]
    work_dirs = [Path(future.command.split()[-1]) for future in futures]

    assert all(work_dir.parent == tmp_path for work_dir in work_dirs)
    assert all(work_dir.name.startswith(prefix) for work_dir in work_dirs)
    assert len(set(work_dirs)) == len(work_dirs)
    assert all(work_dir.is_dir() for work_dir in work_dirs)

    for future in futures:
        future.result()
    assert all(work_dir.is_dir() for work_dir in work_dirs)

    del future
    del futures
    gc.collect()
    assert all(not work_dir.exists() for work_dir in work_dirs)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("affine", [False, True])
def test_search_formats_matrix_and_gap_penalty(seed, affine):
    """
    A custom matrix and affine penalty are converted into MMseqs2 options.
    """
    rng = np.random.default_rng(seed)
    matrix_names = ["BLOSUM45", "BLOSUM62", "PAM30", "PAM250"]
    matrix_name = matrix_names[int(rng.integers(len(matrix_names)))]
    gap_open = -int(rng.integers(5, 20))
    gap_extend = -int(rng.integers(1, 5))
    gap_penalty = (gap_open, gap_extend) if affine else gap_open
    expected_extend = gap_extend if affine else gap_open
    app = MMseqsApp("true")
    database = SequenceDatabase(app)

    future = app.search(
        database,
        database,
        matrix=SubstitutionMatrix(
            ProteinSequence.alphabet,
            ProteinSequence.alphabet,
            matrix_name,
        ),
        gap_penalty=gap_penalty,
    )

    assert f"--gap-open {-gap_open}" in future.command
    assert f"--gap-extend {-expected_extend}" in future.command
    assert "--sub-mat" in future.command
    matrix_path = Path(future.command.split("--sub-mat ", 1)[1].split()[0])
    matrix_text = matrix_path.read_text()
    assert "# Background (precomputed optional):" not in matrix_text
    assert "# Lambda     (precomputed optional):" not in matrix_text
    matrix_lines = [
        line for line in matrix_text.splitlines() if not line.startswith("#")
    ]
    assert matrix_lines[0].split()[-1] == "X"
    assert matrix_lines[-1].split()[0] == "X"
    future.result()


def test_mmseqs_command_pipeline(mmseqs_app, sequences, tmp_path):
    """
    All local MMseqs2 database commands compose through their future results.
    """
    sequence_db = create_database_from_sequences(mmseqs_app, sequences)

    fasta_path = tmp_path / "sequences.fasta"
    assert mmseqs_app.convert_to_fasta(sequence_db, fasta_path).result() == fasta_path
    assert fasta_path.is_file()

    keys_path = tmp_path / "keys.txt"
    keys_path.write_text("0\n")
    mmseqs_app.create_sub_db(keys_path, sequence_db).result()

    alignment_db = mmseqs_app.search(
        sequence_db,
        sequence_db,
        matrix=SubstitutionMatrix.std_protein_matrix(),
        gap_penalty=(-11, -1),
        a=True,
        k=5,
        threads=1,
    ).result()
    alignment_path = tmp_path / "alignments.tsv"
    assert (
        mmseqs_app.convert_alignments(
            alignment_db,
            alignment_path,
            format_output="query,target",
            threads=1,
        ).result()
        == alignment_path
    )
    assert alignment_path.is_file()

    msa_db = mmseqs_app.result_to_msa(alignment_db, threads=1).result()
    profile_db = mmseqs_app.msa_to_profile(msa_db, threads=1).result()
    representative_db = mmseqs_app.profile_to_sequences(profile_db, threads=1).result()
    # The representative sequence of each profile is the query of its MSA
    assert get_sequences_from_database(mmseqs_app, representative_db) == sequences

    profile_alignment_db = mmseqs_app.search(
        profile_db, sequence_db, k=5, threads=1
    ).result()
    assert profile_alignment_db.query_db is profile_db

    linked_path = tmp_path / "linked_database"
    assert mmseqs_app.link_db(sequence_db.name, linked_path).result() == linked_path
    assert linked_path.is_symlink()


def test_foldseek_command_pipeline(foldseek_app, structure_path, tmp_path):
    """
    Foldseek uses the same future and database abstractions for structure inputs.
    """
    sequence_db = foldseek_app.create_db(structure_path, threads=1).result()

    fasta_path = tmp_path / "structure.fasta"
    foldseek_app.convert_to_fasta(sequence_db, fasta_path).result()
    assert fasta_path.is_file()

    alignment_db = foldseek_app.search(
        sequence_db,
        sequence_db,
        a=True,
        threads=1,
    ).result()
    alignment_path = tmp_path / "structure.tsv"
    foldseek_app.convert_alignments(
        alignment_db,
        alignment_path,
        format_output="query,target",
        threads=1,
    ).result()
    assert alignment_path.is_file()


def test_mmseqs_profile_pipeline(mmseqs_app, sequences, tmp_path):
    """
    The profile related commands compose through their future results.
    """
    sequence_db = create_database_from_sequences(mmseqs_app, sequences)
    alignment_db = mmseqs_app.search(sequence_db, sequence_db, k=5, threads=1).result()

    realigned_db = mmseqs_app.align(
        alignment_db, gap_penalty=(-11, -1), threads=1
    ).result()
    assert realigned_db.query_db is sequence_db

    profile_db = mmseqs_app.result_to_profile(realigned_db, threads=1).result()
    consensus_db = mmseqs_app.profile_to_consensus(profile_db, threads=1).result()
    assert len(get_sequences_from_database(mmseqs_app, consensus_db)) == len(sequences)

    pssm_path = tmp_path / "profiles.pssm"
    assert (
        mmseqs_app.profile_to_pssm(profile_db, pssm_path, threads=1).result()
        == pssm_path
    )
    # A header, a column header and one row per position for each profile
    lines = pssm_path.read_text().splitlines()
    assert len(lines) == len(sequences) * (2 + len(sequences["first"]))

    mmseqs_app.search_reciprocal_best_hits(
        sequence_db, sequence_db, k=5, threads=1
    ).result()


@pytest.mark.parametrize("linear", [False, True])
def test_mmseqs_cluster_pipeline(mmseqs_app, sequences, linear, tmp_path):
    """
    Both clustering workflows give a cluster database that can be exported.
    """
    sequence_db = create_database_from_sequences(mmseqs_app, sequences)

    command = mmseqs_app.cluster_linear if linear else mmseqs_app.cluster
    cluster_db = command(sequence_db, threads=1).result()
    assert cluster_db.sequence_db is sequence_db

    tsv_path = tmp_path / "clusters.tsv"
    assert mmseqs_app.create_tsv(cluster_db, tsv_path, threads=1).result() == tsv_path
    members = [line.split("\t")[1] for line in tsv_path.read_text().splitlines()]
    assert sorted(members) == sorted(sequences)


def test_unpack_db(mmseqs_app, sequences, tmp_path):
    """
    Unpacking writes one file per database entry, named by its accession.
    """
    sequence_db = create_database_from_sequences(mmseqs_app, sequences)

    directory = mmseqs_app.unpack_db(
        sequence_db, tmp_path / "unpacked", threads=1
    ).result()

    files = sorted(directory.iterdir())
    assert [file.name for file in files] == sorted(sequences)
    assert {file.name: file.read_text().strip("\n\x00") for file in files} == sequences


def test_foldseek_structure_pipeline(foldseek_app, structure_path, multimer_path):
    """
    The Foldseek specific commands compose through their future results.
    """
    structure_db = foldseek_app.create_db(structure_path, threads=1).result()
    alignment_db = foldseek_app.search(structure_db, structure_db, threads=1).result()

    foldseek_app.structure_align(alignment_db, threads=1).result()
    foldseek_app.search_reciprocal_best_hits(
        structure_db, structure_db, threads=1
    ).result()
    foldseek_app.cluster(structure_db, threads=1).result()

    multimer_db = foldseek_app.create_db(multimer_path, threads=1).result()
    multimer_hits = foldseek_app.search_multimers(
        multimer_db, multimer_db, threads=1
    ).result()
    report_file = foldseek_app.create_multimer_report(multimer_hits, threads=1).result()
    try:
        with open(report_file.name) as file:
            rows = [line.split("\t") for line in file.read().splitlines()]
    finally:
        report_file.close()
    # The self hit of the complex must be reported with a perfect TM-score
    assert float(rows[0][4]) == pytest.approx(1.0)
