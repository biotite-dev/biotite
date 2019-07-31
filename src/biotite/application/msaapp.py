# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["MSAApp"]

import abc
from collections import OrderedDict
import numpy as np
from .localapp import LocalApp
from .application import AppState, requires_state
from ..sequence.sequence import Sequence
from ..sequence.seqtypes import NucleotideSequence, ProteinSequence
from ..sequence.io.fasta.file import FastaFile
from ..sequence.align.alignment import Alignment
from ..temp import temp_file


class MSAApp(LocalApp, metaclass=abc.ABCMeta):
    """
    Perform a multiple sequence alignment.
    
    Internally this creates a `Popen` instance, which handles
    the execution.
    
    Parameters
    ----------
    sequences : iterable object of Sequence
        The sequences to be aligned.
    bin_path : str, optional
        Path of the MSA software binary. By default, the default path
        will be used.
    """
    
    def __init__(self, sequences, bin_path):
        # Check if all sequences share the same alphabet
        alphabet = sequences[0].get_alphabet()
        for seq in sequences:
            if seq.get_alphabet() != alphabet:
                raise ValueError("Alphabets of the sequences are not equal")
        super().__init__(bin_path)
        self._sequences = sequences
        self._in_file_name  = temp_file("fa")
        self._out_file_name = temp_file("fa")

    def run(self):
        in_file = FastaFile()
        for i, seq in enumerate(self._sequences):
            in_file[str(i)] = str(seq)
        in_file.write(self._in_file_name)
        super().run()
    
    def evaluate(self):
        super().evaluate()
        out_file = FastaFile()
        out_file.read(self._out_file_name)
        seq_dict = OrderedDict(out_file)
        # Get alignment
        out_seq_str = [None] * len(seq_dict)
        for i in range(len(self._sequences)):
            out_seq_str[i] = seq_dict[str(i)]
        trace = Alignment.trace_from_strings(out_seq_str)
        self._alignment = Alignment(self._sequences, trace, None)
        # Also obtain original order
        self._order = np.zeros(len(seq_dict), dtype=int)
        for i, seq_index in enumerate(seq_dict):
             self._order[i] = int(seq_index)
    
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
    
    @requires_state(AppState.JOINED)
    def get_alignment_order(self):
        """
        Get the order of the resulting multiple sequence alignment.

        Usually the order of sequences in the output file is
        different from the input file, e.g. the sequences are ordered
        according to the guide tree.
        When using `align()` this order is rearranged so that its is the
        same as the input order. This method returns the original order 
        of the sequences that can be used to restore the MSA software
        intended order.
        
        Returns
        -------
        order : ndarray, dtype=int
            The sequence order intended by the MSA software.
        
        Examples
        --------
        Align sequences and restore the original order:

        app = ClustalOmegaApp(sequences)
        app.start()
        app.join()
        alignment = app.get_alignment()
        order = app.get_alignment_order()
        alignment = alignment[:, order]
        """
        return self._order
    
    def get_input_file_path(self):
        """
        Get input file path (FASTA format).
        
        PROTECTED: Do not call from outside.
        
        Returns
        -------
        path : str
            Path of input file.
        """
        return self._in_file_name
    
    def get_output_file_path(self):
        """
        Get output file path (FASTA format).
        
        PROTECTED: Do not call from outside.
        
        Returns
        -------
        path : str
            Path of output file.
        """
        return self._out_file_name
    
    @classmethod
    def align(cls, sequences, bin_path=None):
        """
        Perform a multiple sequence alignment.
        
        This is a convenience function, that wraps the `MSAApp`
        execution.
        
        Parameters
        ----------
        sequences : iterable object of Sequence
            The sequences to be aligned
        bin_path : str, optional
            Path of the MSA software binary. By default, the default path
            will be used.
        
        Returns
        -------
        alignment : Alignment
            The global multiple sequence alignment.
        """
        app = cls(sequences, bin_path)
        app.start()
        app.join()
        return app.get_alignment()
