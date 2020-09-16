# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from tempfile import TemporaryFile
import biotite.sequence as seq
import biotite.sequence.io.fastq as fastq
import numpy as np
import os
import os.path
from ..util import data_dir
import pytest

@pytest.mark.parametrize("chars_per_line", [None, 80])
def test_access(chars_per_line):
    path = os.path.join(data_dir("sequence"), "random.fastq")
    file = fastq.FastqFile.read(
        path, offset=33, chars_per_line=chars_per_line
    )
    assert len(file) == 20
    assert list(file.keys()) == [f"Read:{i+1:02d}" for i in range(20)]
    del(file["Read:05"])
    assert len(file) == 19
    assert list(file.keys()) == [f"Read:{i+1:02d}" for i in range(20)
                                 if i+1 != 5]
    for seq_str, scores in file.values():
        assert len(seq_str) == len(scores)
        assert (scores >= 0).all()
    seq_str = "ACTCGGT"
    scores = np.array([10,12,20,11,0,80,42])
    file["test"] = seq_str, scores
    seq_str2, scores2 = file["test"]
    assert seq_str == seq_str2
    assert np.array_equal(scores, scores2)

@pytest.mark.parametrize("chars_per_line", [None, 80])
def test_conversion(chars_per_line):
    path = os.path.join(data_dir("sequence"), "random.fastq")
    fasta_file = fastq.FastqFile.read(
        path, offset=33, chars_per_line=chars_per_line
    )
    ref_content = dict(fasta_file.items())

    fasta_file = fastq.FastqFile(offset=33, chars_per_line=chars_per_line)
    for identifier, (sequence, scores) in ref_content.items():
        fasta_file[identifier] = sequence, scores
    temp = TemporaryFile("w+")
    fasta_file.write(temp)

    temp.seek(0)
    fasta_file = fastq.FastqFile.read(
        temp, offset=33, chars_per_line=chars_per_line
    )
    content = dict(fasta_file.items())
    temp.close()
    
    for identifier in ref_content:
        ref_sequence, ref_scores = ref_content[identifier]
        test_sequence, test_scores = content[identifier]
        assert test_sequence == ref_sequence
        assert np.array_equal(test_scores, ref_scores)


def test_rna_conversion():
    sequence = seq.NucleotideSequence("ACGT")
    scores = np.array([0, 0, 0, 0])
    fastq_file = fastq.FastqFile(offset="Sanger")
    fastq.set_sequence(fastq_file, sequence, scores, "seq1", as_rna=False)
    fastq.set_sequence(fastq_file, sequence, scores, "seq2", as_rna=True)
    assert fastq_file["seq1"][0] == "ACGT" 
    assert fastq_file["seq2"][0] == "ACGU"