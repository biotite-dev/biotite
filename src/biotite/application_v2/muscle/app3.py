# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application_v2.muscle"
__author__ = "Patrick Kunzmann"
__all__ = ["Muscle3App", "Muscle3Result"]

import warnings
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from os import PathLike
from tempfile import NamedTemporaryFile
from typing import IO, Any
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
from biotite.application_v2.msa import MSAInput, resolve_gap_penalty
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.phylo.tree import Tree
from biotite.sequence.sequence import Sequence


@dataclass(frozen=True)
class Muscle3Result:
    """
    The result of a MUSCLE version 3 alignment run.

    Attributes
    ----------
    alignment : Alignment
        The global multiple sequence alignment.
    order : ndarray, dtype=int
        The order of the sequences intended by MUSCLE.
        Usually this order (e.g. based on the guide tree) differs from
        the input order.
    guide_tree_kmer : Tree or None
        The guide tree from the first progressive alignment iteration,
        using common *k*-mers as distance measure.
        None, if MUSCLE did not write the tree.
    guide_tree_identity : Tree or None
        The guide tree from the second progressive alignment iteration,
        using distances based on the pairwise sequence identity after
        the first iteration.
        None, if MUSCLE did not write the tree.
    """

    alignment: Alignment
    order: np.ndarray
    guide_tree_kmer: Tree | None
    guide_tree_identity: Tree | None


class Muscle3App(LocalApp):
    """
    A handle to *MUSCLE* version 3.

    Parameters
    ----------
    path : str, optional
        Path of the MUSCLE binary.

    See Also
    --------
    Muscle5App : Interface to MUSCLE version ``>=5``.

    Examples
    --------

    >>> sequences = [
    ...     ProteinSequence("BIQTITE"),
    ...     ProteinSequence("TITANITE"),
    ...     ProteinSequence("BISMITE"),
    ...     ProteinSequence("IQLITE"),
    ... ]
    >>> result = Muscle3App().run(sequences).result()
    >>> print(result.alignment)
    BIQT-ITE
    TITANITE
    BISM-ITE
    -IQL-ITE
    """

    def __init__(self, path: PathLike[str] | str = "muscle") -> None:
        super().__init__(path)
        if self.version.major != 3:
            raise VersionError(f"Muscle 3 is required, got version {self.version}")

    def _format_key(self, key: Any) -> str:
        # MUSCLE 3 uses single-dash options, e.g. '-in' or '-version'
        return "-" + str(key)

    @command(allowed_options=["maxiters", "maxhours", "diags"])
    def run(
        self,
        sequences: SequenceABC[Sequence],
        matrix: SubstitutionMatrix | None = None,
        gap_penalty: float | tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> CommandSetup[Muscle3Result]:
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
        future : Future of Muscle3Result
            A handle to the running alignment.
            Call :meth:`Future.result()` to obtain the
            :class:`Muscle3Result`.
        """
        msa_input = MSAInput.from_input(
            sequences,
            matrix,
            supports_nucleotide=True,
            supports_protein=True,
            supports_custom_nucleotide_matrix=False,
            supports_custom_protein_matrix=True,
        )
        gap_open, gap_ext = resolve_gap_penalty(gap_penalty)

        in_file = NamedTemporaryFile("w", suffix=".fa", delete=False)
        out_file = NamedTemporaryFile("r", suffix=".fa", delete=False)
        matrix_file = NamedTemporaryFile("w", suffix=".mat", delete=False)
        tree1_file = NamedTemporaryFile("r", suffix=".tree", delete=False)
        tree2_file = NamedTemporaryFile("r", suffix=".tree", delete=False)

        msa_input.write_fasta(in_file)
        if msa_input.matrix is not None:
            matrix_file.write(str(msa_input.matrix))
            matrix_file.flush()

        parameters: list[CLIParameter] = [
            CLIFlag("quiet"),
            CLIOption("in", in_file.name),
            CLIOption("out", out_file.name),
            CLIOption("tree1", tree1_file.name),
            CLIOption("tree2", tree2_file.name),
            CLIOption(
                "seqtype", "protein" if msa_input.seqtype == "protein" else "dna"
            ),
        ]
        if msa_input.matrix is not None:
            parameters.append(CLIOption("matrix", matrix_file.name))
        if gap_open is not None:
            parameters += [
                CLIOption("gapopen", f"{gap_open:.1f}"),
                CLIOption("gapextend", f"{gap_ext:.1f}"),
                # When the gap penalty is set,
                # use the penalty also for hydrophobic regions
                CLIOption("hydrofactor", "1.0"),
                # Use the recommendation of the documentation
                CLIOption("center", "0.0"),
            ]

        def evaluate(stdout: bytes, stderr: bytes) -> Muscle3Result:
            alignment, order = msa_input.read_fasta(out_file)
            return Muscle3Result(
                alignment=alignment,
                order=order,
                guide_tree_kmer=_read_tree(tree1_file, "first"),
                guide_tree_identity=_read_tree(tree2_file, "second"),
            )

        def cleanup() -> None:
            for temp_file in (in_file, out_file, matrix_file, tree1_file, tree2_file):
                cleanup_tempfile(temp_file)

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            cleanup=cleanup,
        )


def _read_tree(temp_file: IO[str], iteration: str) -> Tree | None:
    """
    Read a Newick guide tree written by MUSCLE, warning if it is empty.
    """
    newick = temp_file.read().replace("\n", "")
    if len(newick) > 0:
        return Tree.from_newick(newick)
    warnings.warn(f"MUSCLE did not write a tree file from the {iteration} iteration")
    return None
