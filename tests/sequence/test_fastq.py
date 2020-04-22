# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite
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
    for sequence, scores in file.values():
        assert len(sequence) == len(scores)
        assert (scores >= 0).all()
    sequence = seq.NucleotideSequence("ACTCGGT")
    scores = np.array([10,12,20,11,0,80,42])
    file["test"] = sequence, scores
    sequence2, scores2 = file["test"]
    assert sequence == sequence2
    assert np.array_equal(scores, scores2)

@pytest.mark.parametrize("chars_per_line", [None, 80])
def test_conversion(chars_per_line):
    path = os.path.join(data_dir("sequence"), "random.fastq")
    file1 = fastq.FastqFile.read(
        path, offset=33, chars_per_line=chars_per_line
    )
    ref_content = dict(file1.items())

    file2 = fastq.FastqFile(offset=33, chars_per_line=chars_per_line)
    for identifier, (sequence, scores) in ref_content.items():
        file2[identifier] = sequence, scores
    file2.write(biotite.temp_file("fastq"))

    file3 = fastq.FastqFile.read(
        path, offset=33, chars_per_line=chars_per_line
    )
    content = dict(file3.items())
    
    for identifier in ref_content:
        ref_sequence, ref_scores = ref_content[identifier]
        sequence, scores = content[identifier]
        assert ref_sequence == sequence
        assert np.array_equal(ref_scores, scores)