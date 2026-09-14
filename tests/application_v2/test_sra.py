# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
from biotite.application_v2.sra import FastqDumpApp, PrefetchApp
from biotite.sequence.io.fasta import FastaFile
from biotite.sequence.io.fastq import FastqFile
from tests.util import is_not_installed

# Small dataset for low server impact
UID = "ERR11344941"


@pytest.mark.skipif(
    is_not_installed("fasterq-dump"), reason="sra-tools is not installed"
)
@pytest.mark.parametrize("mode", ["fastq", "fasta"])
@pytest.mark.parametrize("use_prefetch", [False, True])
def test_dump(mode, use_prefetch):
    """
    Test the dump application in both modes, with and without a preceding
    prefetch.
    """
    if use_prefetch:
        prefetch = PrefetchApp().run(UID).result()
    else:
        prefetch = None

    app = FastqDumpApp()
    if mode == "fastq":
        result = app.extract_fastq(UID, prefetch).result()
        expected_file_type = FastqFile
    else:
        result = app.extract_fasta(UID, prefetch).result()
        expected_file_type = FastaFile

    # The result stores only the file paths; everything else is lazy
    assert len(result.file_paths) > 0
    for path in result.file_paths:
        assert path.is_file()

    for file in result.get_files():
        assert isinstance(file, expected_file_type)
    for sequences in result.get_sequences():
        assert isinstance(sequences, dict)

    if mode == "fastq":
        for sequences_and_scores in result.get_sequences_and_scores():
            assert isinstance(sequences_and_scores, dict)
