# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from ..localapp import LocalApp
from ...sequence.sequence import Sequence
from ...sequence.seqtypes import NucleotideSequence, ProteinSequence
from ...sequence.io.fasta.file import FastaFile
from ...sequence.align.align import Alignment
from ...temp import temp_file

__all__ = ["MuscleApp"]


_ncbi_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

class MuscleApp(LocalApp):
    
    _counter = 0
    
    def __init__(self, sequences, bin_path="muscle", mute=True):
        super().__init__(bin_path, mute)
        self._sequences = sequences
        MuscleApp._counter += 1
        self._id = MuscleApp._counter

    def run(self):
        in_file_name = temp_file("muscle_in_{:d}.fa".format(self._id))
        out_file_name = temp_file("muscle_out_{:d}.fa".format(self._id))
        in_file = FastaFile()
        for i, seq in enumerate(self._sequences):
            in_file.add(str(i), seq)
        in_file.write(in_file_name)
        self.set_options(["-in", in_file_name, "-out", out_file_name])
        super().run()
