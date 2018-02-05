# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

import biotite.sequence as seq
from biotite.application.muscle import MuscleApp
from biotite.application.mafft import MafftApp
from biotite.application.clustalo import ClustalOmegaApp
import numpy as np
from requests.exceptions import ConnectionError
import pytest
import os.path
import shutil

@pytest.mark.skipif(shutil.which("muscle")   is None or
                    shutil.which("mafft")    is None or
                    shutil.which("clustalo") is None,
                    reason="At least one MSA application is not installed")
@pytest.mark.parametrize("app,exp_ali", [(MuscleApp,       "BIQT-ITE\n"
                                                           "TITANITE\n"
                                                           "BISM-ITE\n"
                                                           "-IQL-ITE"),
                                          
                                         (MafftApp,        "-BIQTITE\n"
                                                           "TITANITE\n"
                                                           "-BISMITE\n"
                                                           "--IQLITE"),
                                         
                                         (ClustalOmegaApp, "-BIQTITE\n"
                                                           "TITANITE\n"
                                                           "-BISMITE\n"
                                                           "--IQLITE")]
                        )
def test_msa(app, exp_ali):
    seq1 = seq.ProteinSequence("BIQTITE")
    seq2 = seq.ProteinSequence("TITANITE")
    seq3 = seq.ProteinSequence("BISMITE")
    seq4 = seq.ProteinSequence("IQLITE")
    alignment = app.align([seq1, seq2, seq3, seq4])
    assert str(alignment) == exp_ali