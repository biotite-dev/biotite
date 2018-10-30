# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["MuscleApp"]

import numbers
from ...temp import temp_file
from ..msaapp import MSAApp
from ..application import AppState, requires_state
from ...sequence.sequence import Sequence
from ...sequence.seqtypes import NucleotideSequence, ProteinSequence
from ...sequence.align.matrix import SubstitutionMatrix
from ...sequence.align.alignment import Alignment


class MuscleApp(MSAApp):
    """
    Perform a multiple sequence alignment using MUSCLE.
    
    Parameters
    ----------
    sequences : iterable object of ProteinSequence
        The sequences to be aligned.
    bin_path : str, optional
        Path of the MUSCLE binary.
    """
    
    def __init__(self, sequences, bin_path=None):
        super().__init__(sequences, bin_path)
        self._matrix_file_name = None
        self._matrix = None
        self._gap_open = None
        self._gap_ext = None
        self._terminal_penalty = None
    
    @requires_state(AppState.CREATED)
    def set_matrix(self, matrix):
        """
        Set the substitution matrix for the alignment.

        Parameters
        ----------
        matix : SubstitutionMatrix
            The substitution matrix to be used.
        """
        self._matrix = matrix
        self._matrix_file_name = temp_file("mat")
    
    @requires_state(AppState.CREATED)
    def set_gap_penalty(self, gap_penalty):
        """
        Set the gap penalty for the alignment.

        Parameters
        ----------
        gap_penalty : float or (tuple, dtype=int), optional
            If a float is provided, the value will be interpreted as
            general gap penalty.
            If a tuple is provided, an affine gap penalty is used.
            The first value in the tuple is the gap opening penalty,
            the second value is the gap extension penalty.
            The values need to be negative.
            """
        # Check if gap penalty is general or affine
        if isinstance(gap_penalty, numbers.Real):
            if gap_penalty > 0:
                raise ValueError("Gap penalty must be negative")
            self._gap_open = gap_penalty
            self._gap_ext= gap_penalty
        elif type(gap_penalty) == tuple:
            if gap_penalty[0] > 0 or gap_penalty[1] > 0:
                    raise ValueError("Gap penalty must be negative")
            self._gap_open = gap_penalty[0]
            self._gap_ext = gap_penalty[1]
        else:
            raise TypeError("Gap penalty must be either float or tuple")
    
    @staticmethod
    def get_default_bin_path():
        return "muscle"
    
    def get_cli_arguments(self):
        args = [
            "-in",  self.get_input_file_path(),
            "-out", self.get_output_file_path()
        ]
        if self._matrix is not None:
            args += ["-matrix", self._matrix_file_name]
        if self._gap_open is not None and self._gap_ext is not None:
            args += ["-gapopen",   f"{self._gap_open:.1f}"]
            args += ["-gapextend", f"{self._gap_ext:.1f}"]
            # When the gap penalty is set,
            # use the penalty also for hydrophobic regions
            args += ["-hydrofactor", "1.0"]
            # Use the recommendation of the documentation
            args += ["-center", "0.0"]
        return args
    
    def run(self):
        if self._matrix is not None:
            with open(self._matrix_file_name, "w") as file:
                file.write(str(self._matrix))
        super().run()
    
    @classmethod
    def align(cls, sequences, bin_path=None, matrix=None,
              gap_penalty=None):
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
        if matrix is not None:
            app.set_matrix(matrix)
        if gap_penalty is not None:
            app.set_gap_penalty(gap_penalty)
        app.start()
        app.join()
        return app.get_alignment()
