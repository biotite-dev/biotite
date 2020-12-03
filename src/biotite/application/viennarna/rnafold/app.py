# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.viennarna.rnafold"
__author__ = "Tom David Müller"
__all__ = ["RNAfoldApp"]

from tempfile import NamedTemporaryFile
from ...localapp import LocalApp, cleanup_tempfile
from ...application import AppState, requires_state
from ....sequence.io.fasta import FastaFile, set_sequence
from ....sequence import NucleotideSequence
from ....structure.dotbracket import base_pairs_from_dot_bracket

class RNAfoldApp(LocalApp):
    """
    Compute the secondary structure of a nucleic acid sequence using
    ViennaRNA's RNAfold software.

    Internally this creates a :class:`Popen` instance, which handles
    the execution.

    Parameters
    ----------
    sequence : NucleotideSequence
        The nucleotide sequence.
    bin_path : str, optional
        Path of the RNAfold binary.

    Examples
    --------

    >>> app = RNAfoldApp(NucleotideSequence("CGACGTAGATGCTAGCTGACTCGATGC"))
    >>> app.start()
    >>> app.join()
    >>> print(app.get_mfe())
    >>> print(app.get_dot_bracket())
    -1.3
    '(((.((((.......)).)))))....'
    """

    def __init__(self, sequence, bin_path="RNAfold"):
        super().__init__(bin_path)
        self._sequence = sequence
        self._in_file  = NamedTemporaryFile("w", suffix=".fa",  delete=False)

    def run(self):
        in_file = FastaFile()
        set_sequence(in_file, self._sequence)
        in_file.write(self._in_file)
        self._in_file.flush()
        self.set_arguments([self._in_file.name])
        super().run()

    def evaluate(self):
        super().evaluate()
        lines = self.get_stdout().split("\n")
        content = lines[2]
        dotbracket, mfe = content.split(" ", maxsplit=1)
        mfe = float(mfe[1:-1])

        self._mfe = mfe
        self._dotbracket = dotbracket

    def clean_up(self):
        super().clean_up()
        cleanup_tempfile(self._in_file)

    @requires_state(AppState.JOINED)
    def get_mfe(self):
        return self._mfe

    @requires_state(AppState.JOINED)
    def get_dot_bracket(self):
        return self._dotbracket

    @requires_state(AppState.JOINED)
    def get_base_pairs(self):
        return base_pairs_from_dot_bracket(self._dotbracket)

    @staticmethod
    def compute_secondary_structure(sequence, bin_path="RNAfold"):
        app = RNAfoldApp(sequence, bin_path)
        app.start()
        app.join()
        return app.get_dot_bracket(), app.get_mfe()
