# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

from ..localapp import LocalApp
from ..application import AppState, requires_state
from ...sequence.sequence import Sequence
from ...sequence.seqtypes import NucleotideSequence, ProteinSequence
from ...sequence.io.fasta.file import FastaFile
from ...sequence.align.alignment import Alignment
from ...temp import temp_file

__all__ = ["MuscleApp"]


_ncbi_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

class MuscleApp(LocalApp):
    """
    Perform a multiple sequence alignment using MUSCLE.
    
    Internally this creates a `Popen` instance, which handles
    the execution.
    
    Parameters
    ----------
    sequences : iterable object of ProteinSequence
        The sequences to be aligned.
    bin_path : str
        Path of the MUCLE binary.
    mute : bool, optional
        If true, the console output goes into DEVNULL. (Default: True)
    """
    
    # Prevents overwriting of input and output files
    # of different MuscleApp instancs
    _counter = 0
    
    def __init__(self, sequences, bin_path="muscle", mute=True):
        super().__init__(bin_path, mute)
        self._sequences = sequences
        MuscleApp._counter += 1
        self._id = MuscleApp._counter

    def run(self):
        in_file_name = temp_file("muscle_in_{:d}.fa".format(self._id))
        self._out_file_name = temp_file("muscle_out_{:d}.fa".format(self._id))
        in_file = FastaFile()
        for i, seq in enumerate(self._sequences):
            in_file[str(i)] = str(seq)
        in_file.write(in_file_name)
        self.set_options(["-in", in_file_name, "-out", self._out_file_name])
        super().run()
    
    def evaluate(self):
        super().evaluate()
        out_file = FastaFile()
        out_file.read(self._out_file_name)
        seq_dict = dict(out_file)
        out_seq_str = [None] * len(seq_dict)
        for i in range(len(self._sequences)):
            out_seq_str[i] = seq_dict[str(i)]
        trace = Alignment.trace_from_strings(out_seq_str)
        self._alignment = Alignment(self._sequences, trace, None)
    
    @requires_state(AppState.JOINED)
    def get_alignment(self):
        """
        Get the resulting multiple sequence alignment.
        
        Returns
        -------
        alignment : Alignment
            The global multiple sequence alignment.
        """
        return self._alignment
