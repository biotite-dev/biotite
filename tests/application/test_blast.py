# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

import biotite.sequence as seq
import biotite.application.blast as blast
import numpy as np
from requests.exceptions import ConnectionError
import pytest
import os.path
from ..sequence.util import data_dir

# Start of E. coli lacZ ORF (UID: AJ308295)
dna_seq = seq.NucleotideSequence("ATGACCATGATTACGCCAAGCTTTCCGGGGAATTCA")

# Start of E. coli lacZ, translated dna_seq (UID: AJ308295)
prot_seq = seq.ProteinSequence("MTMITPSFPGNS")


@pytest.mark.xfail(raises=ConnectionError)
def test_blastn():
    app = blast.BlastWebApp("blastn", dna_seq, obey_rules=False)
    app.set_max_expect_value(100)
    app.start()
    app.join(timeout=300)
    alignments = app.get_alignments()
    # BLAST should find original sequence as best hit
    assert dna_seq == alignments[0].sequences[0]
    assert dna_seq == alignments[0].sequences[1]

@pytest.mark.xfail(raises=ConnectionError)
def test_blastx():
    app = blast.BlastWebApp("blastx", dna_seq, obey_rules=False)
    app.set_max_expect_value(100)
    app.start()
    app.join(timeout=300)
    alignments = app.get_alignments()
    # BLAST should find original sequence as best hit
    assert prot_seq == alignments[0].sequences[0]
    assert prot_seq == alignments[0].sequences[1]

@pytest.mark.xfail(raises=ConnectionError)
def test_tblastx():
    app = blast.BlastWebApp("tblastx", dna_seq, obey_rules=False)
    app.set_max_expect_value(100)
    app.start()
    app.join(timeout=300)
    alignments = app.get_alignments()
    # BLAST should find original sequence as best hit
    print (alignments[0].sequences[0])
    print (alignments[0].sequences[1])
    rev_prot_seq = dna_seq.reverse().complement().translate(complete=True)
    assert rev_prot_seq == alignments[0].sequences[0]
    assert rev_prot_seq == alignments[0].sequences[1]

@pytest.mark.xfail(raises=ConnectionError)
def test_blastp():
    app = blast.BlastWebApp("blastp", prot_seq, obey_rules=False)
    app.set_max_expect_value(100)
    app.start()
    app.join(timeout=300)
    alignments = app.get_alignments()
    # BLAST should find original sequence as best hit
    assert prot_seq == alignments[0].sequences[0]
    assert prot_seq == alignments[0].sequences[1]

@pytest.mark.xfail(raises=ConnectionError)
def test_tblastn():
    app = blast.BlastWebApp("tblastn", prot_seq, obey_rules=False)
    app.set_max_expect_value(200)
    app.start()
    app.join(timeout=300)
    alignments = app.get_alignments()
    # BLAST should find original sequence as best hit
    assert prot_seq == alignments[0].sequences[0]
    assert prot_seq == alignments[0].sequences[1]

def test_file_input():
    path = os.path.join(data_dir, "prot.fasta")
    app = blast.BlastWebApp("blastp", path, obey_rules=False)

def test_invalid_query():
    with pytest.raises(ValueError):
        app = blast.BlastWebApp("blastn", "ABCDEFGHIJKLMNOP", obey_rules=False)
    with pytest.raises(ValueError):
        app = blast.BlastWebApp("blastp", "ABCDEFGHIJKLMNOP", obey_rules=False)
        
@pytest.mark.xfail(raises=ConnectionError)
def test_no_hit():
    app = blast.BlastWebApp("blastn", "ACTGTACGAAACTCGGCGTA", obey_rules=False)
    app.set_word_size(20)
    app.start()
    app.join(timeout=300)
    alignments = app.get_alignments()
    # BLAST should find original sequence as best hit
    assert len(alignments) == 0

@pytest.mark.xfail(raises=ConnectionError)
def test_invalid_input():
    app = blast.BlastWebApp("blastn", dna_seq, obey_rules=False)
    # Set some invalid parameters
    app.set_word_size(-20)
    app.set_substitution_matrix("FOOBAR")
    app.start()
    with pytest.raises(ValueError):
        app.join(timeout=300)




