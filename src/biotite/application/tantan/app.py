# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.tantan"
__author__ = "Patrick Kunzmann"
__all__ = ["TantanApp"]

import io
from collections.abc import Sequence as SequenceABC
from tempfile import NamedTemporaryFile
import numpy as np
from biotite.application.application import AppState, requires_state
from biotite.application.localapp import LocalApp, cleanup_tempfile
from biotite.sequence.alphabet import common_alphabet
from biotite.sequence.io.fasta.file import FastaFile
from biotite.sequence.seqtypes import NucleotideSequence, ProteinSequence

MASKING_LETTER = "!"


class TantanApp(LocalApp):
    r"""
    Mask sequence repeat regions using *tantan*. :footcite:`Frith2011`

    Parameters
    ----------
    sequence : (list of) NucleotideSequence or ProteinSequence
        The sequence(s) to be masked.
        Either a single sequence or multiple sequences can be masked.
        Masking multiple sequences in a single run decreases the
        run time compared to multiple runs with a single sequence.
        All sequences must be of the same type.
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

        if isinstance(sequence, SequenceABC):
            self._as_list = True
            self._sequences = sequence
        else:
            # Convert to list of sequences anyway for consistent handling
            self._as_list = False
            self._sequences = [sequence]

        self._is_protein = None
        for seq in self._sequences:
            if isinstance(seq, NucleotideSequence):
                if self._is_protein is True:
                    # Already protein sequences in the list
                    raise ValueError(
                        "List of sequences contains mixed "
                        "nucleotide and protein sequences"
                    )
                self._is_protein = False
            elif isinstance(seq, ProteinSequence):
                if self._is_protein is False:
                    # Already nucleotide sequences in the list
                    raise ValueError(
                        "List of sequences contains mixed "
                        "nucleotide and protein sequences"
                    )
                self._is_protein = True
            else:
                raise TypeError("A NucleotideSequence or ProteinSequence is required")

        if matrix is None:
            self._matrix_file = None
        else:
            common_alph = common_alphabet((seq.alphabet for seq in self._sequences))
            if common_alph is None:
                raise ValueError("There is no common alphabet within the sequences")
            if not matrix.get_alphabet1().extends(common_alph):
                raise ValueError(
                    "The alphabet of the sequence(s) do not fit the matrix"
                )
            if not matrix.is_symmetric():
                raise ValueError("A symmetric matrix is required")
            self._matrix_file = NamedTemporaryFile("w", suffix=".mat", delete=False)
        self._matrix = matrix

        self._in_file = NamedTemporaryFile("w", suffix=".fa", delete=False)

    def run(self):
        FastaFile.write_iter(
            self._in_file,
            ((f"sequence_{i:d}", str(seq)) for i, seq in enumerate(self._sequences)),
        )
        self._in_file.flush()
        if self._matrix is not None:
            self._matrix_file.write(str(self._matrix))
            self._matrix_file.flush()

        args = []
        if self._matrix is not None:
            args += ["-m", self._matrix_file.name]
        if self._is_protein:
            args += ["-p"]
        args += ["-x", MASKING_LETTER, self._in_file.name]
        self.set_arguments(args)
        super().run()

    def evaluate(self):
        super().evaluate()

        out_file = io.StringIO(self.get_stdout())
        self._masks = []
        encoded_masking_letter = MASKING_LETTER.encode("ASCII")[0]
        for _, masked_seq_string in FastaFile.read_iter(out_file):
            array = np.frombuffer(masked_seq_string.encode("ASCII"), dtype=np.ubyte)
            self._masks.append(array == encoded_masking_letter)

    def clean_up(self):
        super().clean_up()
        cleanup_tempfile(self._in_file)
        if self._matrix_file is not None:
            cleanup_tempfile(self._matrix_file)

    @requires_state(AppState.JOINED)
    def get_mask(self):
        """
        Get a boolean mask covering identified repeat regions of each
        input sequence.

        Returns
        -------
        repeat_mask : (list of) ndarray, shape=(n,), dtype=bool
            A boolean mask that is true for each sequence position that
            is identified as repeat.
            If a list of sequences were given as input, a list of masks
            is returned instead.
        """
        if self._as_list:
            return self._masks
        else:
            return self._masks[0]

    @staticmethod
    def mask_repeats(sequence, matrix=None, bin_path="tantan"):
        """
        Mask repeat regions of the given input sequence(s).

        Parameters
        ----------
        sequence : (list of) NucleotideSequence or ProteinSequence
            The sequence(s) to be masked.
            Either a single sequence or multiple sequences can be masked.
            Masking multiple sequences in a single run decreases the
            run time compared to multiple runs with a single sequence.
            All sequences must be of the same type.
        matrix : SubstitutionMatrix, optional
            The substitution matrix to use for repeat identification.
            A sequence segment is considered to be a repeat of another
            segment, if the substitution score between these segments is
            greater than a threshold value.
        bin_path : str, optional
            Path of the *tantan* binary.

        Returns
        -------
        repeat_mask : (list of) ndarray, shape=(n,), dtype=bool
            A boolean mask that is true for each sequence position that
            is identified as repeat.
            If a list of sequences were given as input, a list of masks
            is returned instead.
        """
        app = TantanApp(sequence, matrix, bin_path)
        app.start()
        app.join()
        return app.get_mask()
