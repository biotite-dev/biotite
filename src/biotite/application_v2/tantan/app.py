# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application_v2.tantan"
__author__ = "Patrick Kunzmann"
__all__ = ["TantanApp"]

import io
from collections.abc import Iterable
from os import PathLike
from tempfile import NamedTemporaryFile
from typing import Any
import numpy as np
from biotite.application_v2.localapp import (
    CLIArgument,
    CLIFlag,
    CLIOption,
    CLIParameter,
    CommandSetup,
    LocalApp,
    cleanup_tempfile,
    command,
)
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.alphabet import common_alphabet
from biotite.sequence.io.fasta.file import FastaFile
from biotite.sequence.seqtypes import NucleotideSequence, ProteinSequence
from biotite.typing import K, NDArray1

MASKING_LETTER = "!"


class TantanApp(LocalApp):
    r"""
    A handle to ``tantan``. :footcite:`Frith2011`

    Parameters
    ----------
    path : str, optional
        Path of the ``tantan`` binary.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> sequence = NucleotideSequence("GGCATCGATATATATATATAGTCAA")
    >>> masks = TantanApp().run([sequence]).result()
    >>> repeat_mask = masks[0]
    >>> print(repeat_mask)
    [False False False False False False False False False  True  True  True
      True  True  True  True  True  True  True  True False False False False
     False]
    >>> print(sequence, "\n" + "".join(["^" if e else " " for e in repeat_mask]))
    GGCATCGATATATATATATAGTCAA
             ^^^^^^^^^^^
    """

    def __init__(self, path: PathLike[str] | str = "tantan") -> None:
        super().__init__(path)

    @command(allowed_options=["r", "e", "w", "d", "s"])
    def run(
        self,
        sequences: Iterable[NucleotideSequence | ProteinSequence],
        matrix: SubstitutionMatrix | None = None,
        **kwargs: Any,
    ) -> CommandSetup[list[NDArray1[K, np.bool_]]]:
        r"""
        Mask sequence repeat regions.

        Parameters
        ----------
        sequences : iterable object of NucleotideSequence or ProteinSequence
            The sequences to be masked.
            Masking multiple sequences in a single run decreases the
            run time compared to multiple runs with a single sequence.
            All sequences must be of the same type.
        matrix : SubstitutionMatrix, optional
            The substitution matrix to use for repeat identification.
            A sequence segment is considered to be a repeat of another
            segment, if the substitution score between these segments is
            greater than a threshold value.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of list of ndarray, shape=(k,), dtype=bool
            A handle to the running masking.
            Call :meth:`Future.result()` to obtain one boolean mask per
            input sequence (in input order), where each mask is true for
            the sequence positions identified as repeat.

        References
        ----------

        .. footbibliography::
        """
        sequences = list(sequences)

        is_protein: bool | None = None
        for seq in sequences:
            if isinstance(seq, NucleotideSequence):
                if is_protein is True:
                    # Already protein sequences in the list
                    raise ValueError(
                        "List of sequences contains mixed "
                        "nucleotide and protein sequences"
                    )
                is_protein = False
            elif isinstance(seq, ProteinSequence):
                if is_protein is False:
                    # Already nucleotide sequences in the list
                    raise ValueError(
                        "List of sequences contains mixed "
                        "nucleotide and protein sequences"
                    )
                is_protein = True
            else:
                raise TypeError("A NucleotideSequence or ProteinSequence is required")

        in_file = NamedTemporaryFile("w", suffix=".fa", delete=False)
        FastaFile.write_iter(
            in_file,
            ((f"sequence_{i:d}", str(seq)) for i, seq in enumerate(sequences)),
        )
        in_file.flush()

        matrix_file = None
        if matrix is not None:
            common_alph = common_alphabet((seq.alphabet for seq in sequences))
            if common_alph is None:
                raise ValueError("There is no common alphabet within the sequences")
            if not matrix.get_alphabet1().extends(common_alph):
                raise ValueError(
                    "The alphabet of the sequence(s) do not fit the matrix"
                )
            if not matrix.is_symmetric():
                raise ValueError("A symmetric matrix is required")
            matrix_file = NamedTemporaryFile("w", suffix=".mat", delete=False)
            matrix_file.write(str(matrix))
            matrix_file.flush()

        parameters: list[CLIParameter] = []
        if matrix_file is not None:
            parameters.append(CLIOption("m", matrix_file.name))
        if is_protein:
            parameters.append(CLIFlag("p"))
        parameters.append(CLIOption("x", MASKING_LETTER))
        parameters.append(CLIArgument(in_file.name))

        def evaluate(stdout: bytes, stderr: bytes) -> list[NDArray1[K, np.bool_]]:
            out_file = io.StringIO(stdout.decode("UTF-8"))
            masks = []
            encoded_masking_letter = MASKING_LETTER.encode("ASCII")[0]
            for _, masked_seq_string in FastaFile.read_iter(out_file):
                array = np.frombuffer(masked_seq_string.encode("ASCII"), dtype=np.ubyte)
                masks.append(array == encoded_masking_letter)
            return masks

        def cleanup() -> None:
            cleanup_tempfile(in_file)
            if matrix_file is not None:
                cleanup_tempfile(matrix_file)

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            cleanup=cleanup,
        )
