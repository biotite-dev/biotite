# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["MafftApp"]

from ..msaapp import MSAApp
from ..application import AppState, requires_state
from ...sequence.sequence import Sequence
from ...sequence.seqtypes import NucleotideSequence, ProteinSequence
from ...sequence.io.fasta.file import FastaFile
from ...sequence.align.alignment import Alignment


class MafftApp(MSAApp):
    """
    Perform a multiple sequence alignment using MAFFT.
    
    Parameters
    ----------
    sequences : iterable object of ProteinSequence or NucleotideSequence
        The sequences to be aligned.
    bin_path : str, optional
        Path of the MUSCLE binary.
    """
    
    def __init__(self, sequences, bin_path="mafft"):
        super().__init__(sequences, bin_path)
    
    def run(self):
        self.set_arguments(
            ["--auto",
             # Get the reordered alignment in order for
             # get_alignment_order() to work properly 
             "--reorder",
             self.get_input_file_path()]
        )
        super().run()
    
    def evaluate(self):
        with open(self.get_output_file_path(), "w") as f:
            # MAFFT outputs alignment to stdout
            # -> write stdout to output file name
            f.write(self.get_stdout())
        super().evaluate()
