# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import glob
import io
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
import numpy as np
import os
import os.path
from ..util import data_dir
import pytest

   
def test_access_low_level():
    path = os.path.join(data_dir("sequence"), "nuc.fasta")
    file = fasta.FastaFile.read(path)
    assert file["dna sequence"] == "ACGCTACGT"
    assert file["another dna sequence"] == "A"
    assert file["third dna sequence"] == "ACGT"
    assert dict(file.items()) == {
        "dna sequence" : "ACGCTACGT",
        "another dna sequence" : "A",
        "third dna sequence" : "ACGT",
        "rna sequence" : "ACGU",
        "ambiguous rna sequence" : "ACGUNN",
    }
    file["another dna sequence"] = "AA"
    del file["dna sequence"]
    file["yet another sequence"] = "ACGT"
    assert dict(file.items()) == {
        "another dna sequence" : "AA",
        "third dna sequence" : "ACGT",
        "rna sequence" : "ACGU",
        "ambiguous rna sequence" : "ACGUNN",
        "yet another sequence" : "ACGT",
    }


def test_access_high_level():
    path = os.path.join(data_dir("sequence"), "nuc.fasta")
    file = fasta.FastaFile.read(path)
    sequences = fasta.get_sequences(file)
    assert sequences == {
        "dna sequence" : seq.NucleotideSequence("ACGCTACGT", False),
        "another dna sequence" : seq.NucleotideSequence("A", False),
        "third dna sequence" : seq.NucleotideSequence("ACGT", False),
        "rna sequence" : seq.NucleotideSequence("ACGT", False),
        "ambiguous rna sequence" : seq.NucleotideSequence("ACGTNN", True),
    }


def test_sequence_conversion():
    path = os.path.join(data_dir("sequence"), "nuc.fasta")
    file = fasta.FastaFile.read(path)
    assert seq.NucleotideSequence("ACGCTACGT") == fasta.get_sequence(file)
    
    seq_dict = fasta.get_sequences(file)
    file2 = fasta.FastaFile()
    fasta.set_sequences(file2, seq_dict)
    seq_dict2 = fasta.get_sequences(file2)
    # Cannot compare dicts directly, since the original RNA sequence is
    # now guessed as protein sequence
    for seq1, seq2 in zip(seq_dict.values(), seq_dict2.values()):
        assert str(seq1) == str(seq2)
    
    file3 = fasta.FastaFile()
    fasta.set_sequence(file3, seq.NucleotideSequence("AACCTTGG"))
    assert file3["sequence"] == "AACCTTGG"
    
    path = os.path.join(data_dir("sequence"), "prot.fasta")
    file4 = fasta.FastaFile.read(path)
    # Expect a warning for selenocysteine conversion
    with pytest.warns(UserWarning):
        assert seq.ProteinSequence("YAHCGFRTGS") == fasta.get_sequence(file4)
    
    path = os.path.join(data_dir("sequence"), "invalid.fasta")
    file5 = fasta.FastaFile.read(path)
    with pytest.raises(ValueError):
        seq.NucleotideSequence(fasta.get_sequence(file5))


def test_rna_conversion():
    sequence = seq.NucleotideSequence("ACGT")
    fasta_file = fasta.FastaFile()
    fasta.set_sequence(fasta_file, sequence, "seq1", as_rna=False)
    fasta.set_sequence(fasta_file, sequence, "seq2", as_rna=True)
    assert fasta_file["seq1"] == "ACGT"
    assert fasta_file["seq2"] == "ACGU"


def test_alignment_conversion():
    path = os.path.join(data_dir("sequence"), "alignment.fasta")
    file = fasta.FastaFile.read(path)
    alignment = fasta.get_alignment(file)
    assert str(alignment) == ("ADTRCGTARDCGTR-DRTCGRAGD\n"
                              "ADTRCGT---CGTRADRTCGRAGD\n"
                              "ADTRCGTARDCGTRADR--GRAGD")
    
    file2 = fasta.FastaFile()
    fasta.set_alignment(file2, alignment, seq_names=["seq1","seq2","seq3"])
    alignment2 = fasta.get_alignment(file2)
    assert str(alignment) == str(alignment2)

@pytest.mark.parametrize(
    "file_name",
    glob.glob(os.path.join(data_dir("sequence"), "*.fasta"))
)
def test_read_iter(file_name):
    ref_dict = dict(fasta.FastaFile.read(file_name).items())
    
    test_dict = dict(fasta.FastaFile.read_iter(file_name))

    assert test_dict == ref_dict


@pytest.mark.parametrize(
    "chars_per_line, n_sequences", itertools.product(
        [80, 200],
        [1, 10]
    )
)
def test_write_iter(chars_per_line, n_sequences):
    """
    Test whether :class:`FastaFile.write()` and
    :class:`FastaFile.write_iter()` produce the same output file for
    random sequences.
    """
    LENGTH_RANGE = (50, 150)
    SCORE_RANGE = (10, 60)

    # Generate random sequences and scores
    np.random.seed(0)
    sequences = []
    for i in range(n_sequences):
        seq_length = np.random.randint(*LENGTH_RANGE)
        code = np.random.randint(
            len(seq.NucleotideSequence.alphabet_unamb),
            size=seq_length
        )
        sequence = seq.NucleotideSequence()
        sequence.code = code
        sequences.append(sequence)
    
    fasta_file = fasta.FastaFile(chars_per_line)
    for i, sequence in enumerate(sequences):
        header = f"seq_{i}"
        fasta_file[header] = str(sequence)
    ref_file = io.StringIO()
    fasta_file.write(ref_file)
    
    test_file = io.StringIO()
    fasta.FastaFile.write_iter(
        test_file,
        (
            (f"seq_{i}", str(sequence))
            for i, sequence in enumerate(sequences)
        ),
        chars_per_line
    )

    assert test_file.getvalue() == ref_file.getvalue()