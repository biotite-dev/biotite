# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.viennarna.rnafold"
__author__ = "Tom David Müller"
__all__ = ["RNAplotApp"]

import numpy as np
from tempfile import NamedTemporaryFile
from os import remove
from ...localapp import LocalApp, cleanup_tempfile
from ...application import AppState, requires_state
from ....sequence.io.fasta import FastaFile, set_sequence
from ....sequence import NucleotideSequence
from ....structure.dotbracket import dot_bracket

class RNAplotApp(LocalApp):

    def __init__(self, dot_bracket = None, base_pairs=None, length = None,
                 bin_path="RNAplot"):
        super().__init__(bin_path)

        if dot_bracket is not None:
            self._dot_bracket = dot_bracket

        elif (base_pairs is not None) and (length is not None):
            self._dot_bracket = dot_bracket(
                base_pairs, length, max_pseudoknot_depth = 0
            )
        else:
            raise ValueError(
                "Structure has to be provided in either dot bracket notation "
                "or as base pairs and total sequence length"
            )

        self._in_file  = NamedTemporaryFile("w", suffix=".fold",  delete=False)

    def run(self):
        self._in_file.write("N"*len(self._dot_bracket) + "\n")
        self._in_file.write(self._dot_bracket)
        self._in_file.flush()
        self.set_arguments(["-i", self._in_file.name, "-o", "xrna"])
        super().run()

    def evaluate(self):
        super().evaluate()
        self._coordinates = np.loadtxt("rna.ss", usecols=(2, 3))

    def clean_up(self):
        super().clean_up()
        cleanup_tempfile(self._in_file)
        remove("rna.ss")

    @requires_state(AppState.JOINED)
    def get_coordinates(self):
        return self._coordinates

    @staticmethod
    def compute_coordinates(dot_bracket = None, base_pairs=None, length = None,
                            bin_path="RNAplot"):
        app = RNAplotApp(dot_bracket=dot_bracket, base_pairs=base_pairs,
                         length=length, bin_path=bin_path)
        app.start()
        app.join()
        return app.get_coordinates()