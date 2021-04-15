# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.tantan"
__author__ = "Patrick Kunzmann"
__all__ = ["TantanApp"]

import io
from tempfile import NamedTemporaryFile
import numpy as np
from ..localapp import LocalApp, cleanup_tempfile
from ..application import AppState, requires_state
from ...sequence.sequence import Sequence
from ...sequence.seqtypes import NucleotideSequence, ProteinSequence
from ...sequence.io.fasta.file import FastaFile
from ...sequence.io.fasta.convert import set_sequence


MASKING_LETTER = "!"


class TantanApp(LocalApp):
    """
    Mask sequence reapeat regions.
    """
    
    def __init__(self, sequence, matrix=None, bin_path="tantan"):
        super().__init__(bin_path)

        if not isinstance(sequence, Sequence):
            raise TypeError("A 'Sequence' object is expected")
        if isinstance(sequence, NucleotideSequence):
            self._is_protein=False
            self._matrix = None
        elif isinstance(sequence, ProteinSequence):
            self._is_protein=True
            self._matrix = None
        else:
            if matrix is None:
                raise ValueError(
                    "A substitution matrix must be provided, "
                    "if neither a protein or nucleotide sequence is handled"
                )
        self._sequence = sequence
        
        if matrix is None:
            self._matrix = None
            self._matrix_file = None
        else:
            if not matrix.get_alphabet1().extends(sequence):
                raise ValueError(
                    "The sequence's alphabet does not fit the matrix"
                )
            if not matrix.is_symmetric():
                raise ValueError("A symmetric matrix is required")
            self._matrix = matrix
            self._matrix_file = NamedTemporaryFile(
                "w", suffix=".mat", delete=False
            )
        
        self._in_file = NamedTemporaryFile("w", suffix=".fa", delete=False)


    def run(self):
        sequence_file = FastaFile()
        set_sequence(sequence_file, self._sequence)
        sequence_file.write(self._in_file)
        self._in_file.flush()
        if self._matrix is not None:
            self._matrix_file.write(str(self._matrix))
            self._matrix_file.flush()
        
        args = []
        if self._matrix is not None:
            args += ["-m", self._matrix_file.name]
        if self._is_protein:
             args += ["-p"]
        args += [
            "-x", MASKING_LETTER,
            self._in_file.name
        ]
        self.set_arguments(args)
        super().run()
    

    def evaluate(self):
        super().evaluate()
        masked_file = FastaFile.read(io.StringIO(self.get_stdout()))
        masked_sequence = masked_file["sequence"]
        array = np.frombuffer(masked_sequence.encode("ASCII"), dtype=np.ubyte)
        self._mask = (array != MASKING_LETTER.encode("ASCII")[0])
    

    def clean_up(self):
        super().clean_up()
        cleanup_tempfile(self._in_file)
        if self._matrix_file is not None:
            cleanup_tempfile(self._matrix_file)
    

    @requires_state(AppState.JOINED)
    def get_mask(self):
        return self._mask


@staticmethod
def mask_repeats(sequence, matrix=None, bin_path="tantan"):
    app = TantanApp(sequence, matrix, bin_path)
    app.start()
    app.join()
    return app.get_mask()