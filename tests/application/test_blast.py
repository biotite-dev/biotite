# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import biopython.sequence as seq
import biopython.application.blast as blast
import numpy as np
from requests.exceptions import ConnectionError
import pytest

#@pytest.mark.skip(reason="")
@pytest.mark.xfail(raises=ConnectionError)
def test_blastp_web():
    prot_seq = seq.ProteinSequence("NLYIQWLKDGGPSSGRPPPS")
    app = blast.BlastWebApp("blastp", prot_seq)
    app.start()
    app.join()
    alignments = app.get_alignments()
    # BLAST should find original sequence as best hit
    assert prot_seq == alignments[0].sequences[0]
    assert prot_seq == alignments[0].sequences[1]