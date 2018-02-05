# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

import biotite.sequence as seq
import biotite.application.muscle as muscle
import numpy as np
from requests.exceptions import ConnectionError
import pytest
import os.path
from ..sequence.util import data_dir
import shutil

@pytest.mark.skipif(shutil.which("muscle") is None,
                    reason="MUSCLE is not installed")
def test_muscle():
    seq1 = seq.ProteinSequence("BIQTITE")
    seq2 = seq.ProteinSequence("TITANITE")
    seq3 = seq.ProteinSequence("BISMITE")
    seq4 = seq.ProteinSequence("IQLITE")
    app = muscle.MuscleApp([seq1, seq2, seq3, seq4], bin_path="muscle")
    app.start()
    app.join()
    alignment = app.get_alignment()
    assert str(alignment) == ("BIQT-ITE\n"
                              "TITANITE\n"
                              "BISM-ITE\n"
                              "-IQL-ITE")