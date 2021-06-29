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
from ..util import map_sequence, map_matrix


MASKING_LETTER = "!"


class TantanApp(LocalApp):
    r"""
    Mask sequence reapeat regions using *tantan*. :footcite:`Frith2011`

    Parameters
    ----------
    sequence : NucleotideSequence or ProteinSequence
        The sequence to be masked.
    matrix : SubstitutionMatrix, optional
        The substitution matrix to use for repeat identification.
        A sequence segment is considered to be a repeat of another
        segment, if the substitution score between these segments is
        greater than a threshold value.
    bin_path : str, optional
        Path of the *tantan* binary.

    References
    ----------
    
    .. footbibliography::

    Examples
    --------

    >>> sequence = NucleotideSequence("GGCATCGATATATATATATAGTCAA")
    >>> app = TantanApp(sequence)
    >>> app.start()
    >>> app.join()
    >>> repeat_mask = app.get_mask()
    >>> print(repeat_mask)
    [False False False False False False False False False  True  True  True
      True  True  True  True  True  True  True  True False False False False
     False]
    >>> print(sequence, "\n" + "".join(["^" if e else " " for e in repeat_mask]))
    GGCATCGATATATATATATAGTCAA 
             ^^^^^^^^^^^         
    """
    
    def __init__(self, sequence, matrix=None, bin_path="tantan"):
        super().__init__(bin_path)

        if not isinstance(sequence, Sequence):
            raise TypeError("A 'Sequence' object is expected")
        
        if matrix is None:
            self._matrix_file = None
        else:
            if not matrix.get_alphabet1().extends(sequence.alphabet):
                raise ValueError(
                    "The sequence's alphabet does not fit the matrix"
                )
            if not matrix.is_symmetric():
                raise ValueError("A symmetric matrix is required")
            self._matrix_file = NamedTemporaryFile(
                "w", suffix=".mat", delete=False
            )
        
        if isinstance(sequence, NucleotideSequence):
            self._is_protein=False
            self._sequence = sequence
            self._matrix = matrix
        elif isinstance(sequence, ProteinSequence):
            self._is_protein=True
        else:
            raise TypeError(
                "A NucleotideSequence or ProteinSequence is required"
            )
        self._sequence = sequence
        self._matrix = matrix
        
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
        self._mask = (array == MASKING_LETTER.encode("ASCII")[0])
    

    def clean_up(self):
        super().clean_up()
        cleanup_tempfile(self._in_file)
        if self._matrix_file is not None:
            cleanup_tempfile(self._matrix_file)
    

    @requires_state(AppState.JOINED)
    def get_mask(self):
        """
        Get a boolean mask covering identified repeat regions of the
        input sequence.

        Returns
        -------
        repeat_mask : ndarray, shape=(n,), dtype=bool
            A boolean mask that is true for each sequence position that
            is identified as repeat.
        """
        return self._mask


    @staticmethod
    def mask_repeats(sequence, matrix=None, bin_path="tantan"):
        """
        Mask repeat regions of the given input sequence.

        Parameters
        ----------
        sequence : NucleotideSequence or ProteinSequence
            The sequence to be masked.
        matrix : SubstitutionMatrix, optional
            The substitution matrix to use for repeat identification.
            A sequence segment is considered to be a repeat of another
            segment, if the substitution score between these segments is
            greater than a threshold value.
        bin_path : str, optional
            Path of the *tantan* binary.

        Returns
        -------
        repeat_mask : ndarray, shape=(n,), dtype=bool
            A boolean mask that is true for each sequence position that
            is identified as repeat.
        """
        app = TantanApp(sequence, matrix, bin_path)
        app.start()
        app.join()
        return app.get_mask()