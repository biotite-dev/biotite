# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application_v2.mafft"
__author__ = "Patrick Kunzmann"
__all__ = ["MafftApp", "MafftResult"]

import os
import re
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from io import StringIO
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
from biotite.application_v2.msa import MSAInput, resolve_gap_penalty
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.phylo.tree import Tree
from biotite.sequence.sequence import Sequence

# MAFFT labels each leaf of the guide tree as '<n>_<sequence name>'
_PREFIX_PATTERN = re.compile(r"\d*_")


@dataclass(frozen=True)
class MafftResult:
    """
    The result of a MAFFT alignment run.

    Attributes
    ----------
    alignment : Alignment
        The global multiple sequence alignment.
    order : ndarray, dtype=int
        The order of the sequences intended by MAFFT.
        Usually this order (e.g. based on the guide tree) differs from
        the input order.
    guide_tree : Tree
        The guide tree created for the progressive alignment.
    """

    alignment: Alignment
    order: np.ndarray
    guide_tree: Tree


class MafftApp(LocalApp):
    """
    A handle to *MAFFT*.

    Parameters
    ----------
    path : str, optional
        Path of the MAFFT binary.

    Examples
    --------

    >>> sequences = [
    ...     ProteinSequence("BIQTITE"),
    ...     ProteinSequence("TITANITE"),
    ...     ProteinSequence("BISMITE"),
    ...     ProteinSequence("IQLITE"),
    ... ]
    >>> result = MafftApp().run(sequences).result()
    >>> print(result.alignment)
    -BIQTITE
    TITANITE
    -BISMITE
    --IQLITE
    """

    def __init__(self, path: PathLike[str] | str = "mafft") -> None:
        super().__init__(path)

    @command(allowed_options=["thread"])
    def run(
        self,
        sequences: SequenceABC[Sequence],
        matrix: SubstitutionMatrix | None = None,
        gap_penalty: float | tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> CommandSetup[MafftResult]:
        """
        Perform a multiple sequence alignment.

        Parameters
        ----------
        sequences : iterable object of Sequence
            The sequences to be aligned.
        matrix : SubstitutionMatrix, optional
            A custom substitution matrix.
        gap_penalty : float or tuple of (float, float), optional
            If a float is provided, the value will be interpreted as
            general gap penalty.
            If a tuple is provided, an affine gap penalty is used.
            The first value in the tuple is the gap opening penalty,
            the second value is the gap extension penalty.
            The values need to be negative.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of MafftResult
            A handle to the running alignment.
            Call :meth:`Future.result()` to obtain the
            :class:`MafftResult`.
        """
        msa_input = MSAInput.from_input(
            sequences,
            matrix,
            supports_nucleotide=True,
            supports_protein=True,
            supports_custom_nucleotide_matrix=True,
            supports_custom_protein_matrix=True,
        )

        in_file = NamedTemporaryFile("w", suffix=".fa", delete=False)
        matrix_file = NamedTemporaryFile("w", suffix=".mat", delete=False)
        # MAFFT writes the guide tree next to the input file
        tree_file_name = in_file.name + ".tree"
        msa_input.write_fasta(in_file)
        if msa_input.matrix is not None:
            matrix_file.write(str(msa_input.matrix))
            matrix_file.flush()

        parameters: list[CLIParameter] = [
            CLIFlag("quiet"),
            CLIFlag("auto"),
            CLIFlag("treeout"),
            # Reorder the output for `order` to reflect the guide tree
            CLIFlag("reorder"),
            CLIFlag("amino" if msa_input.seqtype == "protein" else "nuc"),
            # The input file is a positional argument
            CLIArgument(in_file.name),
        ]
        if msa_input.matrix is not None:
            parameters.append(CLIOption("aamatrix", matrix_file.name))
        gap_open, gap_ext = resolve_gap_penalty(gap_penalty)
        if gap_open is not None and gap_ext is not None:
            # MAFFT expects the gap penalties as positive magnitudes
            parameters += [
                CLIOption("op", f"{abs(gap_open):g}"),
                CLIOption("ep", f"{abs(gap_ext):g}"),
            ]

        def evaluate(stdout: bytes, stderr: bytes) -> MafftResult:
            # MAFFT writes the alignment to the standard output
            alignment, order = msa_input.read_fasta(StringIO(stdout.decode("UTF-8")))
            with open(tree_file_name) as file:
                raw_newick = file.read().replace("\n", "")
            # Remove the '<n>_' prefix from each leaf label
            newick = re.sub(_PREFIX_PATTERN, "", raw_newick)
            return MafftResult(
                alignment=alignment,
                order=order,
                guide_tree=Tree.from_newick(newick),
            )

        def cleanup() -> None:
            for temp_file in (in_file, matrix_file):
                cleanup_tempfile(temp_file)
            try:
                os.remove(tree_file_name)
            except FileNotFoundError:
                pass

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            cleanup=cleanup,
        )
