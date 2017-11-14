# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

import biotite
import biotite.database.entrez as entrez
import biotite.sequence.io.fasta as fasta
import numpy as np
from requests.exceptions import ConnectionError
import pytest


@pytest.mark.xfail(raises=ConnectionError)
def test_fetch():
    file = entrez.fetch("1L2Y_A", biotite.temp_dir(), "fa", "protein",
                        "fasta", overwrite=True)
    fasta_file = fasta.FastaFile()
    fasta_file.read(file)
    prot_seq = fasta.get_sequence(fasta_file)

@pytest.mark.xfail(raises=ConnectionError)
def test_fetch_invalid():
    with pytest.raises(ValueError):
        file = entrez.fetch("xxxx", biotite.temp_dir(), "fa", "protein",
                            "fasta", overwrite=True)