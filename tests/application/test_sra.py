# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
from os.path import join
from tempfile import gettempdir
import pytest
from biotite.application.sra import FastaDumpApp, FastqDumpApp
from biotite.sequence.io.fasta import FastaFile
from biotite.sequence.io.fastq import FastqFile
from tests.util import is_not_installed


@pytest.mark.skipif(
    is_not_installed("fasterq-dump"), reason="sra-tools is not installed"
)
@pytest.mark.parametrize(
    "app_class, custom_prefix",
    itertools.product([FastqDumpApp, FastaDumpApp], [False, True]),
)
def test_objects(app_class, custom_prefix):
    """
    Test return types of methods from the respective `Application` class.
    """
    # Small dataset for low server impact
    UID = "ERR11344941"

    prefix = join(gettempdir(), "test_fastq") if custom_prefix else None
    app = app_class(UID, output_path_prefix=prefix)
    app.start()
    app.join()

    for sequences in app.get_sequences():
        assert isinstance(sequences, dict)

    if app_class == FastqDumpApp:
        for sequences_and_scores in app.get_sequences_and_scores():
            assert isinstance(sequences_and_scores, dict)

    if app_class == FastqDumpApp:
        for fastq_file in app.get_fastq():
            assert isinstance(fastq_file, FastqFile)
    else:
        for fasta_file in app.get_fasta():
            assert isinstance(fasta_file, FastaFile)


@pytest.mark.skipif(
    is_not_installed("fasterq-dump"), reason="sra-tools is not installed"
)
@pytest.mark.parametrize(
    "app_class, custom_prefix",
    itertools.product([FastqDumpApp, FastaDumpApp], [False, True]),
)
def test_classmethod(app_class, custom_prefix):
    """
    Test return types of the `fetch()` class method.
    """
    UID = "ERR11344941"

    prefix = join(gettempdir(), "test_fastq") if custom_prefix else None

    sequences = app_class.fetch(UID, output_path_prefix=prefix)

    for sequences in sequences:
        assert isinstance(sequences, dict)
