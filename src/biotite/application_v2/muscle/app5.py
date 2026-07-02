# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application_v2.muscle"
__author__ = "Patrick Kunzmann"
__all__ = ["Muscle5App", "Muscle5Result"]

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from os import PathLike
from tempfile import NamedTemporaryFile
from typing import Any
import numpy as np
from biotite.application_v2.base import VersionError
from biotite.application_v2.localapp import (
    CLIFlag,
    CLIOption,
    CLIParameter,
    CommandSetup,
    LocalApp,
    cleanup_tempfile,
    command,
)
from biotite.application_v2.msa import MSAInput
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.sequence import Sequence


@dataclass(frozen=True)
class Muscle5Result:
    """
    The result of a MUSCLE version 5 alignment run.

    Attributes
    ----------
    alignment : Alignment
        The global multiple sequence alignment.
    order : ndarray, dtype=int
        The order of the sequences intended by MUSCLE.
        Usually this order (e.g. based on the guide tree) differs from
        the input order.
    """

    alignment: Alignment
    order: np.ndarray


class Muscle5App(LocalApp):
    """
    A handle to *MUSCLE* version 5.

    Parameters
    ----------
    path : str, optional
        Path of the MUSCLE binary.

    See Also
    --------
    Muscle3App : Interface to MUSCLE version ``<5``.

    Notes
    -----
    Alignment ensemble generation is not supported, yet.

    Examples
    --------

    >>> sequences = [
    ...     ProteinSequence("BIQTITE"),
    ...     ProteinSequence("TITANITE"),
    ...     ProteinSequence("BISMITE"),
    ...     ProteinSequence("IQLITE"),
    ... ]
    >>> result = Muscle5App().run(sequences).result()
    >>> print(result.alignment)
    BI-QTITE
    TITANITE
    BI-SMITE
    -I-QLITE
    """

    def __init__(self, path: PathLike[str] | str = "muscle") -> None:
        super().__init__(path)
        if self.version.major < 5:
            raise VersionError(
                f"At least Muscle 5 is required, got version {self.version}"
            )

    def _format_key(self, key: Any) -> str:
        # MUSCLE 5 uses single-dash options, e.g. '-align' or '-version'
        return "-" + str(key)

    @command(allowed_options=["threads", "consiters", "refineiters", "perturb"])
    def run(
        self,
        sequences: SequenceABC[Sequence],
        super5: bool = False,
        **kwargs: Any,
    ) -> CommandSetup[Muscle5Result]:
        """
        Perform a multiple sequence alignment.

        Parameters
        ----------
        sequences : iterable object of Sequence
            The sequences to be aligned.
        super5 : bool, optional
            If set, the *Super5* algorithm is used for the alignment,
            which is faster for a large number of sequences.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of Muscle5Result
            A handle to the running alignment.
            Call :meth:`Future.result()` to obtain the
            :class:`Muscle5Result`.
        """
        msa_input = MSAInput.from_input(
            sequences,
            None,
            supports_nucleotide=True,
            supports_protein=True,
            supports_custom_nucleotide_matrix=False,
            supports_custom_protein_matrix=False,
        )

        in_file = NamedTemporaryFile("w", suffix=".fa", delete=False)
        out_file = NamedTemporaryFile("r", suffix=".fa", delete=False)
        msa_input.write_fasta(in_file)

        # The alignment mode takes the input file as its value
        mode = "super5" if super5 else "align"
        parameters: list[CLIParameter] = [
            CLIOption(mode, in_file.name),
            CLIOption("output", out_file.name),
            CLIFlag("amino" if msa_input.seqtype == "protein" else "nt"),
        ]

        def evaluate(stdout: bytes, stderr: bytes) -> Muscle5Result:
            alignment, order = msa_input.read_fasta(out_file)
            return Muscle5Result(alignment=alignment, order=order)

        def cleanup() -> None:
            for temp_file in (in_file, out_file):
                cleanup_tempfile(temp_file)

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            cleanup=cleanup,
        )
