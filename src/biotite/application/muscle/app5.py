# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.muscle"
__author__ = "Patrick Kunzmann"
__all__ = ["Muscle5App"]

import numbers
import warnings
from tempfile import NamedTemporaryFile
from ..localapp import cleanup_tempfile
from ..msaapp import MSAApp
from ..application import AppState, VersionError, requires_state
from ...sequence.sequence import Sequence
from ...sequence.seqtypes import NucleotideSequence, ProteinSequence
from ...sequence.align.matrix import SubstitutionMatrix
from ...sequence.align.alignment import Alignment
from ...sequence.phylo.tree import Tree
from .app3 import get_version


class Muscle5App(MSAApp):
    """
    Perform a multiple sequence alignment using MUSCLE version 5.
    
    Parameters
    ----------
    sequences : list of Sequence
        The sequences to be aligned.
    bin_path : str, optional
        Path of the MUSCLE binary.
    
    See also
    ---------
    MuscleApp

    Notes
    -----
    Alignment ensemble generation is not supported, yet.
    
    Examples
    --------

    >>> seq1 = ProteinSequence("BIQTITE")
    >>> seq2 = ProteinSequence("TITANITE")
    >>> seq3 = ProteinSequence("BISMITE")
    >>> seq4 = ProteinSequence("IQLITE")
    >>> app = Muscle5App([seq1, seq2, seq3, seq4])
    >>> app.start()
    >>> app.join()
    >>> alignment = app.get_alignment()
    >>> print(alignment)
    BI-QTITE
    TITANITE
    BI-SMITE
    -I-QLITE
    """
    
    def __init__(self, sequences, bin_path="muscle"):
        major_version = get_version(bin_path)[0]
        if major_version < 5:
            raise VersionError(
                f"At least Muscle 5 is required, got version {major_version}"
            )
        
        super().__init__(sequences, bin_path)
        self._mode = "align"
        self._consiters = None
        self._refineiters = None
        self._n_threads = None

    @requires_state(AppState.CREATED)
    def set_iterations(self, consistency=None, refinement=None):
        """
        Set the number of iterations for the alignment algorithm.

        Parameters
        ----------
        consistency : int, optional
            The number of consistency iterations.
        refinement : int, optional
            The number of refinement iterations.
        """
        if consistency is not None:
            self._consiters = consistency
        if refinement is not None:
            self._refineiters = refinement
    
    @requires_state(AppState.CREATED)
    def set_thread_number(self, number):
        """
        Set the number of threads for the alignment run.

        Parameters
        ----------
        number : int, optional
            The number of threads.
        """
        self._n_threads = number

    @requires_state(AppState.CREATED)
    def use_super5(self):
        """
        Use the *Super5* algorithm for the alignment run.
        """
        self._mode = "super5"

    def run(self):
        args = [
            f"-{self._mode}",
            self.get_input_file_path(),
            "-output", self.get_output_file_path(),
        ]
        if self.get_seqtype() == "protein":
            args += ["-amino"]
        else:
            args += ["-nt"]
        if self._n_threads is not None:
             args += ["-threads", str(self._n_threads)]
        if self._consiters is not None:
             args += ["-consiters", str(self._consiters)]
        if self._refineiters is not None:
             args += ["-refineiters", str(self._refineiters)]
        self.set_arguments(args)
        super().run()
    
    def clean_up(self):
        super().clean_up()
    
    @staticmethod
    def supports_nucleotide():
        return True
    
    @staticmethod
    def supports_protein():
        return True
    
    @staticmethod
    def supports_custom_nucleotide_matrix():
        return False
    
    @staticmethod
    def supports_custom_protein_matrix():
        return False
    
    @classmethod
    def align(cls, sequences, bin_path=None):
        """
        Perform a multiple sequence alignment.
        
        This is a convenience function, that wraps the :class:`Muscle5App`
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
