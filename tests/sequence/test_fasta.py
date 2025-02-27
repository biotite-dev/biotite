# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
import io
import itertools
import os
import os.path
import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
from tests.util import data_dir


def test_access_low_level():
    path = os.path.join(data_dir("sequence"), "nuc.fasta")
    file = fasta.FastaFile.read(path)
    assert file["dna sequence"] == "ACGCTACGT"
    assert file["another dna sequence"] == "A"
    assert file["third dna sequence"] == "ACGT"
    assert dict(file.items()) == {
        "dna sequence": "ACGCTACGT",
        "another dna sequence": "A",
        "third dna sequence": "ACGT",
        "rna sequence": "ACGU",
        "ambiguous rna sequence": "ACGUNN",
    }
    file["another dna sequence"] = "AA"
    del file["dna sequence"]
    file["yet another sequence"] = "ACGT"
    assert dict(file.items()) == {
        "another dna sequence": "AA",
        "third dna sequence": "ACGT",
        "rna sequence": "ACGU",
        "ambiguous rna sequence": "ACGUNN",
        "yet another sequence": "ACGT",
    }


@pytest.mark.parametrize("seq_type", (None, seq.NucleotideSequence))
def test_access_high_level(seq_type):
    path = os.path.join(data_dir("sequence"), "nuc.fasta")
    file = fasta.FastaFile.read(path)
    sequences = fasta.get_sequences(file, seq_type=seq_type)
    assert sequences == {
        "dna sequence": seq.NucleotideSequence("ACGCTACGT", False),
        "another dna sequence": seq.NucleotideSequence("A", False),
        "third dna sequence": seq.NucleotideSequence("ACGT", False),
        "rna sequence": seq.NucleotideSequence("ACGT", False),
        "ambiguous rna sequence": seq.NucleotideSequence("ACGTNN", True),
    }


@pytest.mark.parametrize(
    "seq_type", (None, seq.NucleotideSequence, seq.ProteinSequence)
)
def test_sequence_conversion_ambiguous(seq_type):
    path = os.path.join(data_dir("sequence"), "nuc.fasta")
    file = fasta.FastaFile.read(path)
    sequence = "ACGCTACGT"

    if seq_type is None:
        # By default a nucleotide sequence is guessed
        assert seq.NucleotideSequence(sequence) == fasta.get_sequence(
            file, seq_type=None
        )
    else:
        assert seq_type(sequence) == fasta.get_sequence(file, seq_type=seq_type)

    seq_dict = fasta.get_sequences(file)
    file2 = fasta.FastaFile()
    fasta.set_sequences(file2, seq_dict)
    seq_dict2 = fasta.get_sequences(file2, seq_type=seq_type)

    if seq_type == seq.ProteinSequence:
        # Cannot compare dicts directly as nucleotide sequence is automatically
        #  guessed
        assert seq_dict != seq_dict2
        for seq1, seq2 in zip(seq_dict.values(), seq_dict2.values()):
            assert str(seq1) == str(seq2)
    else:
        assert seq_dict == seq_dict2

    if seq_type is not None:
        sequence = "AACCTTGG"
        file3 = fasta.FastaFile()
        fasta.set_sequence(file3, seq_type(sequence))
        assert file3["sequence"] == sequence


@pytest.mark.parametrize(
    "seq_type", (None, seq.NucleotideSequence, seq.ProteinSequence)
)
def test_sequence_conversion_protein(seq_type):
    path = os.path.join(data_dir("sequence"), "prot.fasta")
    file4 = fasta.FastaFile.read(path)

    if seq_type != seq.NucleotideSequence:
        # Expect a warning for selenocysteine conversion
        with pytest.warns(UserWarning):
            assert seq.ProteinSequence("YAHCGFRTGS") == fasta.get_sequence(
                file4, seq_type=seq_type
            )
    else:
        # Manually forcing a NucleotideSequence should raise an error
        with pytest.raises(seq.AlphabetError):
            fasta.get_sequence(file4, seq_type=seq_type)


@pytest.mark.parametrize(
    "seq_type", (None, seq.NucleotideSequence, seq.ProteinSequence)
)
def test_sequence_conversion_invalid(seq_type):
    path = os.path.join(data_dir("sequence"), "invalid.fasta")
    file = fasta.FastaFile.read(path)
    with pytest.raises(ValueError):
        seq.NucleotideSequence(fasta.get_sequence(file), seq_type=seq_type)


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
    assert str(alignment) == (
        "ADTRCGTARDCGTR-DRTCGRAGD\n"
        "ADTRCGT---CGTRADRTCGRAGD\n"
        "ADTRCGTARDCGTRADR--GRAGD"
    )  # fmt: skip

    file2 = fasta.FastaFile()
    fasta.set_alignment(file2, alignment, seq_names=["seq1", "seq2", "seq3"])
    alignment2 = fasta.get_alignment(file2)
    assert str(alignment) == str(alignment2)


@pytest.mark.parametrize(
    "file_name", glob.glob(os.path.join(data_dir("sequence"), "*.fasta"))
)
def test_read_iter(file_name):
    ref_dict = dict(fasta.FastaFile.read(file_name).items())

    test_dict = dict(fasta.FastaFile.read_iter(file_name))

    assert test_dict == ref_dict


@pytest.mark.parametrize(
    "chars_per_line, n_sequences", itertools.product([80, 200], [1, 10])
)
def test_write_iter(chars_per_line, n_sequences):
    """
    Test whether :class:`FastaFile.write()` and
    :class:`FastaFile.write_iter()` produce the same output file for
    random sequences.
    """
    LENGTH_RANGE = (50, 150)

    # Generate random sequences and scores
    np.random.seed(0)
    sequences = []
    for i in range(n_sequences):
        seq_length = np.random.randint(*LENGTH_RANGE)
        code = np.random.randint(
            len(seq.NucleotideSequence.alphabet_unamb), size=seq_length
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
        ((f"seq_{i}", str(sequence)) for i, sequence in enumerate(sequences)),
        chars_per_line,
    )

    assert test_file.getvalue() == ref_file.getvalue()
