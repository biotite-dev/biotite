# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import biopython.sequence as seq
import biopython.application.muscle as muscle
import numpy as np
from requests.exceptions import ConnectionError
import pytest
import os.path
from ..sequence.util import data_dir
import shutil

@pytest.mark.skipif(shutil.which("muscle") is None,
                    reason="MUSCLE is not installed")
def test_muscle():
    seq1 = seq.ProteinSequence("BIQPYTHQN")
    seq2 = seq.ProteinSequence("PYLQN")
    seq3 = seq.ProteinSequence("BIQPSY")
    app = muscle.MuscleApp([seq1, seq2, seq3], bin_path="muscle")
    app.start()
    app.join()
    alignment = app.get_alignment()
    assert str(alignment) == "BIQPYTHQN\n---PYL-QN\nBIQPSY---"