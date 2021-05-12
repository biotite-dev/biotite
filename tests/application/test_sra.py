# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
from tempfile import gettempdir
import pytest
from biotite.application.sra import FastqDumpApp
from biotite.sequence.io.fastq import FastqFile


@pytest.mark.parametrize("custom_prefix", [False, True])
def test_fastq_dump(custom_prefix):
    prefix = join(gettempdir(), "test_fastq") if custom_prefix else None
    app = FastqDumpApp("SRR6986517", output_path_prefix=prefix)
    app.start()
    app.join()

    for sequences_and_scores in app.get_sequences():
        assert isinstance(sequences_and_scores, dict)
    
    for fastq_file in app.get_fastq():
        assert isinstance(fastq_file, FastqFile)
