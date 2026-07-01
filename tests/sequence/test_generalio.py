# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
import biotite.sequence.io as seqio
from tests.util import data_dir


@pytest.mark.parametrize(
    "path",
    sorted(data_dir("sequence").glob("random.*")),
    ids=lambda path: path.name,
)
def test_loading_single(path):
    ref_sequence = seqio.load_sequence(data_dir("sequence") / "random.fasta")
    sequence = seqio.load_sequence(path)
    assert ref_sequence == sequence


@pytest.mark.parametrize("suffix", ["fasta", "fastq"])
def test_saving_single(suffix, tmp_path):
    ref_sequence = seqio.load_sequence(data_dir("sequence") / "random.fasta")
    try:
        seqio.save_sequence(tmp_path / f"test.{suffix}", ref_sequence)
    except PermissionError:
        # This error might occur on AppVeyor
        pytest.skip("Permission is denied")


@pytest.mark.parametrize(
    "path",
    sorted(data_dir("sequence").glob("random.*")),
    ids=lambda path: path.name,
)
def test_loading_multiple(path):
    ref_sequences = seqio.load_sequences(data_dir("sequence") / "random.fasta")
    sequences = seqio.load_sequences(path)
    assert ref_sequences == sequences


@pytest.mark.parametrize("suffix", ["fasta", "fastq"])
def test_saving_multiple(suffix, tmp_path):
    ref_sequences = seqio.load_sequences(data_dir("sequence") / "random.fasta")
    try:
        seqio.save_sequences(tmp_path / f"test.{suffix}", ref_sequences)
    except PermissionError:
        # This error might occur on AppVeyor
        pytest.skip("Permission is denied")


@pytest.mark.parametrize("file_name", ["gg_avidin.gb", "bt_lysozyme.gp"])
def test_genbank(file_name, tmp_path):
    """
    Simply test whether reading or writing a GenBank/GenPept file
    raises an exception.
    """
    sequence = seqio.load_sequence(data_dir("sequence") / file_name)
    try:
        seqio.save_sequence(tmp_path / "test.gb", sequence)
    except PermissionError:
        # This error might occur on AppVeyor
        pytest.skip("Permission is denied")
