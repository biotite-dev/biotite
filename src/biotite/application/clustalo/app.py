# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["ClustalOmegaApp"]

from ..msaapp import MSAApp
from ..application import AppState, requires_state
from ...sequence.sequence import Sequence
from ...sequence.seqtypes import NucleotideSequence, ProteinSequence
from ...sequence.io.fasta.file import FastaFile
from ...sequence.align.alignment import Alignment


class ClustalOmegaApp(MSAApp):
    """
    Perform a multiple sequence alignment using Clustal-Omega.
    
    Parameters
    ----------
    sequences : iterable object of ProteinSequence or NucleotideSequence
        The sequences to be aligned.
    bin_path : str, optional
        Path of the Custal-Omega binary.
    """
    
    def __init__(self, sequences, bin_path=None):
        if isinstance(sequences[0], NucleotideSequence):
            self._seqtype = "DNA"
        else:
            self._seqtype = "Protein"
        super().__init__(sequences, bin_path)
    
    @staticmethod
    def get_default_bin_path():
        return "clustalo"
    
    def get_cli_arguments(self):
        return ["--in", self.get_input_file_path(),
                "--out", self.get_output_file_path(),
                "--seqtype", self._seqtype,
                "--output-order=tree-order"]
