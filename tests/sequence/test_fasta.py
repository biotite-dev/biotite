# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
import numpy as np
import os
import os.path
from ..util import data_dir
import pytest

   
def test_access():
    path = os.path.join(data_dir("sequence"), "nuc.fasta")
    file = fasta.FastaFile()
    file.read(path)
    assert file["dna sequence"] == "ACGCTACGT"
    assert file["another dna sequence"] == "A"
    assert file["third dna sequence"] == "ACGT"
    assert dict(file.items()) == {
        "dna sequence" : "ACGCTACGT",
        "another dna sequence" : "A",
        "third dna sequence" : "ACGT"
    }
    file["another dna sequence"] = "AA"
    del file["dna sequence"]
    file["yet another sequence"] = "ACGT"
    assert dict(file.items()) == {
        "another dna sequence" : "AA",
        "third dna sequence"   : "ACGT",
        "yet another sequence" : "ACGT"
    }

def test_sequence_conversion():
    path = os.path.join(data_dir("sequence"), "nuc.fasta")
    file = fasta.FastaFile()
    file.read(path)
    assert seq.NucleotideSequence("ACGCTACGT") == fasta.get_sequence(file)
    
    seq_dict = fasta.get_sequences(file)
    file2 = fasta.FastaFile()
    fasta.set_sequences(file2, seq_dict)
    seq_dict2 = fasta.get_sequences(file2)
    assert seq_dict == seq_dict2
    
    file3 = fasta.FastaFile()
    fasta.set_sequence(file3, seq.NucleotideSequence("AACCTTGG"))
    assert file3["sequence"] == "AACCTTGG"
    
    path = os.path.join(data_dir("sequence"), "prot.fasta")
    file4 = fasta.FastaFile()
    file4.read(path)
    assert seq.ProteinSequence("YAHGFRTGS") == fasta.get_sequence(file4)
    
    path = os.path.join(data_dir("sequence"), "invalid.fasta")
    file5 = fasta.FastaFile()
    file5.read(path)
    with pytest.raises(ValueError):
        seq.NucleotideSequence(fasta.get_sequence(file5))

def test_alignment_conversion():
    path = os.path.join(data_dir("sequence"), "alignment.fasta")
    file = fasta.FastaFile()
    file.read(path)
    alignment = fasta.get_alignment(file)
    assert str(alignment) == ("ADTRCGTARDCGTR-DRTCGRAGD\n"
                              "ADTRCGT---CGTRADRTCGRAGD\n"
                              "ADTRCGTARDCGTRADR--GRAGD")
    
    file2 = fasta.FastaFile()
    fasta.set_alignment(file2, alignment, seq_names=["seq1","seq2","seq3"])
    alignment2 = fasta.get_alignment(file2)
    assert str(alignment) == str(alignment2)
