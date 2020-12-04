# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.viennarna.rnafold"
__author__ = "Tom David Müller"
__all__ = ["RNAplotApp"]

from tempfile import NamedTemporaryFile
from ...localapp import LocalApp, cleanup_tempfile
from ...application import AppState, requires_state
from ....sequence.io.fasta import FastaFile, set_sequence
from ....sequence import NucleotideSequence
from ....structure.dotbracket import base_pairs_from_dot_bracket

class RNAplotApp(LocalApp):

    def __init__(self, dot_bracket = None, base_pairs=None, length = None,
                 bin_path="RNAplot"):
        super().__init__(bin_path)
        # TODO
        self._in_file  = NamedTemporaryFile("w", suffix=".fold",  delete=False)

    def run(self):
        # TODO
        super().run()

    def evaluate(self):
        # TODO
        super().evaluate()

    def clean_up(self):
        super().clean_up()
        # TODO
        cleanup_tempfile(self._in_file)

    @requires_state(AppState.JOINED)
    def get_coordinates(self):
        # TODO
        pass

    @staticmethod
    def compute_coordinates(dot_bracket = None, base_pairs=None, length = None,
                            bin_path="RNAplot"):
        # TODO
        app = RNAplotApp(dot_bracket=dot_bracket, base_pairs=base_pairs,
                         length=length, bin_path=bin_path)
        app.start()
        app.join()
        return app.get_coordinates()