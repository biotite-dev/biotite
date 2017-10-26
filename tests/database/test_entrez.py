# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import biopython
import biopython.database.entrez as entrez
import biopython.sequence.io.fasta as fasta
import numpy as np
from requests.exceptions import ConnectionError
import pytest


@pytest.mark.xfail(raises=ConnectionError)
def test_fetch():
    file = entrez.fetch("1L2Y_A", biopython.temp_dir(), "fa", "protein",
                        "fasta", overwrite=True)
    fasta_file = fasta.FastaFile()
    fasta_file.read(file)
    prot_seq = fasta.get_sequence(fasta_file)

@pytest.mark.xfail(raises=ConnectionError)
def test_fetch_invalid():
    with pytest.raises(ValueError):
        file = entrez.fetch("xxxx", biopython.temp_dir(), "fa", "protein",
                            "fasta", overwrite=True)