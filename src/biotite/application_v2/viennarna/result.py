# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application_v2.viennarna"
__author__ = "Patrick Kunzmann"
__all__ = ["FoldedRNA", "CofoldedRNA", "FoldedAlignment", "CofoldedAlignment"]

from dataclasses import dataclass
import numpy as np
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.seqtypes import NucleotideSequence
from biotite.structure.dotbracket import base_pairs_from_dot_bracket
from biotite.structure.dotbracket import dot_bracket as dot_bracket_
from biotite.typing import C2, C4, K, N, NDArray1, NDArray2

# Canonical (Watson-Crick and wobble) base pairs in both orders, with 'T'
# standing in for 'U', as :class:`NucleotideSequence` uses DNA letters
_CANONICAL_PAIRS = ("AT", "TA", "GC", "CG", "GT", "TG")


def _filter_base_pairs(
    base_pairs: NDArray2[N, C2, np.integer], symbols: NDArray1[K, np.str_]
) -> NDArray1[N, np.bool_]:
    """
    Get a mask marking the base pairs that form a canonical (Watson-Crick
    or wobble) pair for the given symbols.

    Parameters
    ----------
    base_pairs : ndarray, shape=(n,2), dtype=int
        The base pairs as positions into `symbols`.
    symbols : ndarray, shape=(l,), dtype=str
        The nucleotide symbols; gap positions are marked by ``-``.

    Returns
    -------
    mask : ndarray, shape=(n,), dtype=bool
        True for each canonical base pair.
    """
    pair_symbols = np.char.add(symbols[base_pairs[:, 0]], symbols[base_pairs[:, 1]])
    return np.isin(pair_symbols, _CANONICAL_PAIRS)


@dataclass(frozen=True)
class FoldedRNA:
    """
    The predicted secondary structure of a single ribonucleic acid sequence.

    Attributes
    ----------
    sequence : NucleotideSequence
        The folded RNA sequence.
    dot_bracket : str
        The secondary structure in dot-bracket notation.
    free_energy : float, optional
        The free energy (kcal/mol) of the structure.
    span : tuple (int, int), optional
        If the ``dot_bracket`` describes only a local part of the
        sequence, the ``(start, stop)`` range (0-based, exclusive stop)
        the structure occupies in `sequence`.
        ``None`` means the structure spans the entire sequence.
    """

    sequence: NucleotideSequence
    dot_bracket: str
    free_energy: float | None = None
    span: tuple[int, int] | None = None

    def base_pairs(self) -> NDArray2[N, C2, np.integer]:
        """
        Get the base pairs of the structure.

        Returns
        -------
        base_pairs : ndarray, shape=(n,2), dtype=int
            Each row gives the positions of the two paired bases in `sequence`.
        """
        start = 0 if self.span is None else self.span[0]
        return _single_strand_base_pairs(self.dot_bracket, start)

    @staticmethod
    def from_base_pairs(
        sequence: NucleotideSequence,
        base_pairs: NDArray2[N, C2, np.integer],
        max_pseudoknot_order: int | None = None,
    ) -> FoldedRNA:
        """
        Create a :class:`FoldedRNA` from a sequence and its base pairs.

        Non-canonical base pairs are omitted, as they cannot be
        represented by the folding energy model.

        Parameters
        ----------
        sequence : NucleotideSequence
            The RNA sequence.
        base_pairs : ndarray, shape=(n,2), dtype=int
            Each row gives the positions of two paired bases in
            `sequence`.
        max_pseudoknot_order : int, optional
            The maximum pseudoknot order to represent.
            Base pairs of a higher order are treated as unpaired.
            By default, all orders are represented.

        Returns
        -------
        folded : FoldedRNA
            The corresponding structure, with unknown ``free_energy``.
        """
        base_pairs = np.asarray(base_pairs)
        base_pairs = base_pairs[
            _filter_base_pairs(base_pairs, np.asarray(sequence.symbols))
        ]
        notation = dot_bracket_(
            base_pairs, len(sequence), max_pseudoknot_order=max_pseudoknot_order
        )[0]
        return FoldedRNA(sequence, notation)


@dataclass(frozen=True)
class CofoldedRNA:
    """
    The predicted secondary structure of a complex of multiple
    ribonucleic acid strands.

    Attributes
    ----------
    sequences : list of NucleotideSequence
        The strands the complex is composed of.
    dot_bracket : str
        The secondary structure in dot-bracket notation with strands
        separated by ``&``.
    free_energy : float, optional
        The free energy (kcal/mol) of the complex.
    spans : list of tuple (int, int), optional
        If the ``dot_bracket`` describes only local parts of the
        strands, the ``(start, stop)`` range (0-based, exclusive stop)
        each segment occupies in the respective sequence.
        ``None`` means each segment spans its entire sequence.
    """

    sequences: list[NucleotideSequence]
    dot_bracket: str
    free_energy: float | None = None
    spans: list[tuple[int, int]] | None = None

    def base_pairs(self) -> NDArray2[N, C4, np.integer]:
        """
        Get the base pairs of the complex.

        Returns
        -------
        base_pairs : ndarray, shape=(n,4), dtype=int
            Each row is ``(strand_i, pos_i, strand_j, pos_j)``, where the
            strand indices refer to `sequences` and the positions to the
            respective sequence.
        """
        return _multi_strand_base_pairs(self.dot_bracket, self.spans)

    @classmethod
    def from_single(cls, single: FoldedRNA) -> CofoldedRNA:
        """
        Create a :class:`CofoldedRNA` from a single-strand
        :class:`FoldedRNA`.

        Parameters
        ----------
        single : FoldedRNA
            The single-strand structure.

        Returns
        -------
        cofolded : CofoldedRNA
            The equivalent complex containing a single strand.
        """
        return cls(
            sequences=[single.sequence],
            dot_bracket=single.dot_bracket,
            free_energy=single.free_energy,
            spans=None if single.span is None else [single.span],
        )


@dataclass(frozen=True)
class FoldedAlignment:
    """
    The predicted consensus secondary structure of an alignment of
    ribonucleic acid sequences.

    Attributes
    ----------
    alignment : Alignment
        The folded alignment.
    dot_bracket : str
        The consensus secondary structure in dot-bracket notation,
        indexed by alignment column.
    free_energy : float, optional
        The physical free energy (kcal/mol) of the consensus structure.
    covariance_energy : float, optional
        The energy (kcal/mol) of the covariance term, if reported by the
        software.
    span : tuple (int, int), optional
        If the ``dot_bracket`` describes only a local part of the
        alignment, the ``(start, stop)`` range (0-based, exclusive stop)
        of alignment columns the structure occupies.
        ``None`` means the structure spans all columns.
    """

    alignment: Alignment
    dot_bracket: str
    free_energy: float | None = None
    covariance_energy: float | None = None
    span: tuple[int, int] | None = None

    def base_pairs(self) -> NDArray2[N, C2, np.integer]:
        """
        Get the consensus base pairs in alignment-column coordinates.

        Returns
        -------
        base_pairs : ndarray, shape=(n,2), dtype=int
            Each row gives the two paired alignment columns.
        """
        start = 0 if self.span is None else self.span[0]
        return _single_strand_base_pairs(self.dot_bracket, start)

    def base_pairs_of(self, sequence_index: int) -> NDArray2[N, C2, np.integer]:
        """
        Get the consensus base pairs projected onto one sequence of the
        alignment.

        Base pairs where either column is a gap in the selected sequence
        are omitted.

        Parameters
        ----------
        sequence_index : int
            The index of the sequence in the alignment.

        Returns
        -------
        base_pairs : ndarray, shape=(n,2), dtype=int
            Each row gives the two paired positions in the ungapped
            sequence numbering of the selected sequence.
        """
        trace = self.alignment.trace[:, sequence_index]
        residues = trace[self.base_pairs()]
        # Remove base pairs pointing to gaps in the selected sequence
        return residues[np.all(residues != -1, axis=1)].reshape(-1, 2)  # pyright: ignore[reportReturnType]

    @staticmethod
    def from_base_pairs(
        alignment: Alignment,
        base_pairs: NDArray2[N, C2, np.integer],
        max_pseudoknot_order: int | None = None,
    ) -> FoldedAlignment:
        """
        Create a :class:`FoldedAlignment` from an alignment and its
        consensus base pairs.

        Base pairs that are non-canonical in every sequence of the
        alignment are omitted, as they cannot be represented by the
        folding energy model.

        Parameters
        ----------
        alignment : Alignment
            The alignment.
        base_pairs : ndarray, shape=(n,2), dtype=int
            Each row gives the two paired alignment columns.
        max_pseudoknot_order : int, optional
            The maximum pseudoknot order to represent.
            Base pairs of a higher order are treated as unpaired.
            By default, all orders are represented.

        Returns
        -------
        folded : FoldedAlignment
            The corresponding consensus structure, with unknown
            ``free_energy``.
        """
        base_pairs = np.asarray(base_pairs)
        # Keep a base pair if it is canonical in at least one sequence
        mask = np.zeros(len(base_pairs), dtype=bool)
        for gapped_sequence in alignment.get_gapped_sequences():
            mask |= _filter_base_pairs(base_pairs, np.array(list(gapped_sequence)))
        base_pairs = base_pairs[mask]
        notation = dot_bracket_(
            base_pairs, len(alignment), max_pseudoknot_order=max_pseudoknot_order
        )[0]
        return FoldedAlignment(alignment, notation)


@dataclass(frozen=True)
class CofoldedAlignment:
    """
    The predicted consensus secondary structure of a complex of multiple
    ribonucleic acid alignments.

    Attributes
    ----------
    alignments : list of Alignment
        The alignments the complex is composed of.
    dot_bracket : str
        The consensus secondary structure in dot-bracket notation with
        strands separated by ``&``, indexed by alignment column.
    free_energy : float, optional
        The free energy (kcal/mol) of the complex.
    covariance_energy : float, optional
        The energy (kcal/mol) of the covariance term, if reported
        separately by the software.
    spans : list of tuple (int, int), optional
        If the ``dot_bracket`` describes only local parts of the
        alignments, the ``(start, stop)`` range (0-based, exclusive stop)
        of alignment columns each segment occupies.
        ``None`` means each segment spans all columns.
    """

    alignments: list[Alignment]
    dot_bracket: str
    free_energy: float | None = None
    covariance_energy: float | None = None
    spans: list[tuple[int, int]] | None = None

    def base_pairs(self) -> NDArray2[N, C4, np.integer]:
        """
        Get the consensus base pairs in alignment-column coordinates.

        Returns
        -------
        base_pairs : ndarray, shape=(n,4), dtype=int
            Each row is ``(strand_i, col_i, strand_j, col_j)`` with the
            columns referring to the respective alignment.
        """
        return _multi_strand_base_pairs(self.dot_bracket, self.spans)

    def base_pairs_of(self, sequence_index: int) -> NDArray2[N, C4, np.integer]:
        """
        Get the consensus base pairs projected onto one sequence of each
        alignment.

        The same `sequence_index` is applied to every alignment.
        Base pairs where either column is a gap in the selected sequence
        are omitted.

        Parameters
        ----------
        sequence_index : int
            The index of the sequence, applied to every alignment.

        Returns
        -------
        base_pairs : ndarray, shape=(n,4), dtype=int
            Each row is ``(strand_i, pos_i, strand_j, pos_j)`` with the
            positions given in the ungapped numbering of the selected
            sequence of each alignment.
        """
        consensus_pairs = self.base_pairs()
        strands_i, columns_i, strands_j, columns_j = consensus_pairs.T
        positions_i = np.empty(len(consensus_pairs), dtype=int)
        positions_j = np.empty(len(consensus_pairs), dtype=int)
        for strand, alignment in enumerate(self.alignments):
            trace = alignment.trace[:, sequence_index]
            mask_i = strands_i == strand
            positions_i[mask_i] = trace[columns_i[mask_i]]
            mask_j = strands_j == strand
            positions_j[mask_j] = trace[columns_j[mask_j]]
        base_pairs = np.stack([strands_i, positions_i, strands_j, positions_j], axis=-1)
        # Remove base pairs pointing to gaps in the selected sequence
        return base_pairs[(positions_i != -1) & (positions_j != -1)]

    @classmethod
    def from_single(cls, single: FoldedAlignment) -> CofoldedAlignment:
        """
        Create a :class:`CofoldedAlignment` from a single-alignment
        :class:`FoldedAlignment`.

        Parameters
        ----------
        single : FoldedAlignment
            The single-alignment structure.

        Returns
        -------
        cofolded : CofoldedAlignment
            The equivalent complex containing a single alignment.
        """
        return cls(
            alignments=[single.alignment],
            dot_bracket=single.dot_bracket,
            free_energy=single.free_energy,
            covariance_energy=single.covariance_energy,
            spans=None if single.span is None else [single.span],
        )


def _single_strand_base_pairs(
    dot_bracket: str, start: int = 0
) -> NDArray2[N, C2, np.integer]:
    """
    Extract base pairs from a single (``&``-free) dot-bracket string.

    Parameters
    ----------
    dot_bracket : str
        The dot-bracket string.
        May contain pseudoknot brackets (``[](){}<>`` and letters).
    start : int, optional
        An offset added to each position, e.g. to map a local structure
        onto the full sequence.

    Returns
    -------
    base_pairs : ndarray, shape=(n,2), dtype=int
        Each row gives the positions of the two paired bases.
    """
    return base_pairs_from_dot_bracket(dot_bracket) + start


def _multi_strand_base_pairs(
    dot_bracket: str, spans: list[tuple[int, int]] | None = None
) -> NDArray2[N, C4, np.integer]:
    """
    Extract inter- and intra-strand base pairs from a ``&``-separated
    dot-bracket string.

    Parameters
    ----------
    dot_bracket : str
        The dot-bracket string, with strands separated by ``&``.
    spans : list of tuple (int, int), optional
        For each strand the ``(start, stop)`` range the dot-bracket
        segment occupies in the respective full sequence.
        Only the start is used as offset.

    Returns
    -------
    base_pairs : ndarray, shape=(n,4), dtype=int
        Each row is ``(strand_i, pos_i, strand_j, pos_j)`` with the
        opening partner first.
        Rows are sorted canonically.
    """
    segments = dot_bracket.split("&")
    boundaries = np.cumsum([0] + [len(segment) for segment in segments])
    starts = np.array(
        [spans[i][0] if spans is not None else 0 for i in range(len(segments))]
    )
    # Base pairs in coordinates of the concatenated string without '&'
    concat_pairs = base_pairs_from_dot_bracket("".join(segments))
    open_positions = concat_pairs[:, 0]
    close_positions = concat_pairs[:, 1]

    # The strand a position belongs to is the segment its concatenated index
    # falls into; the position within that strand is the offset from the segment
    # start plus the strand's own start offset
    strands_i = np.searchsorted(boundaries, open_positions, side="right") - 1
    strands_j = np.searchsorted(boundaries, close_positions, side="right") - 1
    positions_i = open_positions - boundaries[strands_i] + starts[strands_i]
    positions_j = close_positions - boundaries[strands_j] + starts[strands_j]
    base_pairs = np.stack([strands_i, positions_i, strands_j, positions_j], axis=-1)
    if len(base_pairs) > 0:
        base_pairs = base_pairs[
            np.lexsort(
                (base_pairs[:, 3], base_pairs[:, 2], base_pairs[:, 1], base_pairs[:, 0])
            )
        ]
    return base_pairs  # pyright: ignore[reportReturnType]
