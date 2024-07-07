# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
from os.path import join
from tempfile import NamedTemporaryFile
import pytest
import biotite.sequence.io as seqio
from tests.util import data_dir


@pytest.mark.parametrize("path", glob.glob(join(data_dir("sequence"), "random.*")))
def test_loading_single(path):
    ref_sequence = seqio.load_sequence(join(data_dir("sequence"), "random.fasta"))
    sequence = seqio.load_sequence(path)
    assert ref_sequence == sequence


@pytest.mark.parametrize("suffix", ["fasta", "fastq"])
def test_saving_single(suffix):
    ref_sequence = seqio.load_sequence(join(data_dir("sequence"), "random.fasta"))
    temp = NamedTemporaryFile("w+", suffix=f".{suffix}")
    try:
        seqio.save_sequence(temp.name, ref_sequence)
    except PermissionError:
        # This error might occur on AppVeyor
        pytest.skip("Permission is denied")


@pytest.mark.parametrize("path", glob.glob(join(data_dir("sequence"), "random.*")))
def test_loading_multiple(path):
    ref_sequences = seqio.load_sequences(join(data_dir("sequence"), "random.fasta"))
    sequences = seqio.load_sequences(path)
    assert ref_sequences == sequences


@pytest.mark.parametrize("suffix", ["fasta", "fastq"])
def test_saving_multiple(suffix):
    ref_sequences = seqio.load_sequences(join(data_dir("sequence"), "random.fasta"))
    temp = NamedTemporaryFile("w+", suffix=f".{suffix}")
    try:
        seqio.save_sequences(temp.name, ref_sequences)
    except PermissionError:
        # This error might occur on AppVeyor
        pytest.skip("Permission is denied")


@pytest.mark.parametrize("file_name", ["gg_avidin.gb", "bt_lysozyme.gp"])
def test_genbank(file_name):
    """
    Simply test whether reading or writing a GenBank/GenPept file
    raises an exception.
    """

    temp = NamedTemporaryFile("w+", suffix=".gb")
    sequence = seqio.load_sequence(join(data_dir("sequence"), file_name))
    try:
        seqio.save_sequence(temp.name, sequence)
    except PermissionError:
        # This error might occur on AppVeyor
        pytest.skip("Permission is denied")
