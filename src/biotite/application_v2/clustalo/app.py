# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application_v2.clustalo"
__author__ = "Patrick Kunzmann"
__all__ = ["ClustalOmegaApp", "ClustalOmegaResult"]

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from os import PathLike
from tempfile import NamedTemporaryFile
from typing import Any
import numpy as np
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
from biotite.sequence.phylo.tree import Tree
from biotite.sequence.sequence import Sequence


@dataclass(frozen=True)
class ClustalOmegaResult:
    """
    The result of a Clustal-Omega alignment run.

    Attributes
    ----------
    alignment : Alignment
        The global multiple sequence alignment.
    order : ndarray, dtype=int
        The order of the sequences intended by Clustal-Omega.
        Usually this order (e.g. based on the guide tree) differs from
        the input order.
    guide_tree : Tree
        The guide tree used for the progressive alignment.
        If a guide tree was given as input, this is that tree.
    distance_matrix : ndarray, shape=(n,n), dtype=float or None
        The pairwise sequence distances used to calculate the guide
        tree.
        Only available if ``use_full_matrix`` was set, otherwise None.
    """

    alignment: Alignment
    order: np.ndarray
    guide_tree: Tree
    distance_matrix: np.ndarray | None


class ClustalOmegaApp(LocalApp):
    """
    A handle to *Clustal-Omega*.

    Parameters
    ----------
    path : str, optional
        Path of the Clustal-Omega binary.

    Examples
    --------

    >>> sequences = [
    ...     ProteinSequence("BIQTITE"),
    ...     ProteinSequence("TITANITE"),
    ...     ProteinSequence("BISMITE"),
    ...     ProteinSequence("IQLITE"),
    ... ]
    >>> result = ClustalOmegaApp().run(sequences).result()
    >>> print(result.alignment)
    -BIQTITE
    TITANITE
    -BISMITE
    --IQLITE
    """

    def __init__(self, path: PathLike[str] | str = "clustalo") -> None:
        super().__init__(path)

    @command(allowed_options=["iterations", "threads"])
    def run(
        self,
        sequences: SequenceABC[Sequence],
        distance_matrix: np.ndarray | None = None,
        guide_tree: Tree | None = None,
        use_full_matrix: bool = False,
        **kwargs: Any,
    ) -> CommandSetup[ClustalOmegaResult]:
        """
        Perform a multiple sequence alignment.

        Parameters
        ----------
        sequences : iterable object of Sequence
            The sequences to be aligned.
        distance_matrix : ndarray, shape=(n,n), dtype=float, optional
            Pairwise sequence distances used to calculate the guide tree.
        guide_tree : Tree, optional
            The guide tree used for the progressive alignment.
        use_full_matrix : bool, optional
            If set, the full distance matrix is used for the guide-tree
            calculation, equivalent to the ``--full`` option, instead of
            the default *mBed* heuristic.
            This is required to obtain the distance matrix in the result.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of ClustalOmegaResult
            A handle to the running alignment.
            Call :meth:`Future.result()` to obtain the
            :class:`ClustalOmegaResult`.
        """
        msa_input = MSAInput.from_input(
            sequences,
            None,
            supports_nucleotide=True,
            supports_protein=True,
            supports_custom_nucleotide_matrix=False,
            supports_custom_protein_matrix=False,
        )
        seq_count = len(msa_input.original_sequences)
        if distance_matrix is not None and distance_matrix.shape != (
            seq_count,
            seq_count,
        ):
            raise ValueError(
                f"Distance matrix with shape {distance_matrix.shape} is not "
                f"sufficient for {seq_count} sequences"
            )
        if guide_tree is not None and len(guide_tree) != seq_count:
            raise ValueError(
                f"Guide tree with {len(guide_tree)} leaves is not sufficient "
                f"for {seq_count} sequences"
            )

        in_file = NamedTemporaryFile("w", suffix=".fa", delete=False)
        out_file = NamedTemporaryFile("r", suffix=".fa", delete=False)
        in_matrix_file = NamedTemporaryFile("w", suffix=".mat", delete=False)
        out_matrix_file = NamedTemporaryFile("r", suffix=".mat", delete=False)
        in_tree_file = NamedTemporaryFile("w", suffix=".tree", delete=False)
        out_tree_file = NamedTemporaryFile("r", suffix=".tree", delete=False)
        msa_input.write_fasta(in_file)

        parameters: list[CLIParameter] = [
            CLIOption("in", in_file.name),
            CLIOption("out", out_file.name),
            CLIOption(
                "seqtype", "Protein" if msa_input.seqtype == "protein" else "DNA"
            ),
            # The temporary output files already exist -> overwrite them
            CLIFlag("force"),
            # Tree order for `order` to reflect the guide tree
            CLIOption("output-order", "tree-order"),
        ]
        if guide_tree is None:
            # Clustal-Omega does not accept a tree as input and output
            # at the same time
            parameters.append(CLIOption("guidetree-out", out_tree_file.name))
        if use_full_matrix:
            parameters.append(CLIFlag("full"))
            parameters.append(CLIOption("distmat-out", out_matrix_file.name))
        if distance_matrix is not None:
            # Prepend the sequence indices as first column
            matrix_with_index = np.concatenate(
                (
                    np.arange(seq_count)[:, np.newaxis],
                    distance_matrix.astype(float, copy=False),
                ),
                axis=1,
            )
            np.savetxt(
                in_matrix_file.name,
                matrix_with_index,
                comments="",
                # The first line contains the number of sequences
                header=str(seq_count),
                # The sequence indices are integers, the rest are floats
                fmt=["%d"] + ["%.5f"] * seq_count,
            )
            parameters.append(CLIOption("distmat-in", in_matrix_file.name))
        if guide_tree is not None:
            in_tree_file.write(str(guide_tree))
            in_tree_file.flush()
            parameters.append(CLIOption("guidetree-in", in_tree_file.name))

        def evaluate(stdout: bytes, stderr: bytes) -> ClustalOmegaResult:
            alignment, order = msa_input.read_fasta(out_file)
            if guide_tree is None:
                result_tree = Tree.from_newick(out_tree_file.read().replace("\n", ""))
            else:
                result_tree = guide_tree
            if use_full_matrix:
                # The first row only contains the number of sequences and
                # the first column only the sequence indices
                result_matrix = np.loadtxt(
                    out_matrix_file.name, skiprows=1, dtype=float
                )[:, 1:]
            else:
                result_matrix = None
            return ClustalOmegaResult(
                alignment=alignment,
                order=order,
                guide_tree=result_tree,
                distance_matrix=result_matrix,
            )

        def cleanup() -> None:
            for temp_file in (
                in_file,
                out_file,
                in_matrix_file,
                out_matrix_file,
                in_tree_file,
                out_tree_file,
            ):
                cleanup_tempfile(temp_file)

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            cleanup=cleanup,
        )
