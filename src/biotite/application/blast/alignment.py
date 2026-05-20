# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application.blast"
__author__ = "Patrick Kunzmann"
__all__ = ["BlastAlignment"]

import numpy as np
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.sequence import Sequence
from biotite.typing import K, N, NDArray2


class BlastAlignment(Alignment):
    """
    A specialized :class:`Alignment` class for alignments using the
    BLAST application. It stores additional data, like the E-value,
    the HSP position and a description of the hit sequence.

    Like its superclass, all attributes of a :class:`BlastAlignment` are
    public. The attributes are the same as the constructor parameters.

    Parameters
    ----------
    sequences : list
        A list of aligned sequences. Does actually not contain the
        complete original sequences, but the HSP sequences.
    trace : ndarray, dtype=int, shape=(n,m)
        The alignment trace.
    score : int
        Alignment score.
    e_value : float
        Expectation value for the number of random sequences of a
        similar sized database getting an equal or higher score by
        change when aligned with the query sequence.
    query_interval : tuple of int
        Describes the position of the HSP part of the query sequence
        in the original query sequence. The first element is the start
        position, the second element is the inclusive stop position.
        Indexing starts at 1.
    hit_interval : tuple of int
        Analogous to `query_interval`, this describes the position of
        the HSP part of the hit sequence in the complete hit sequence.
    hit_id : str
        The NCBI *unique identifier* (UID) of the hit sequence.
    hit_definition : str
        The name of the hit sequence.
    """

    e_value: float
    query_interval: tuple[int, int]
    hit_interval: tuple[int, int]
    hit_id: str
    hit_definition: str

    def __init__(
        self,
        sequences: list[Sequence],
        trace: NDArray2[K, N, np.integer],
        score: int,
        e_value: float,
        query_interval: tuple[int, int],
        hit_interval: tuple[int, int],
        hit_id: str,
        hit_definition: str,
    ) -> None:
        super().__init__(sequences, trace, score)
        self.e_value = e_value
        self.query_interval = query_interval
        self.hit_interval = hit_interval
        self.hit_id = hit_id
        self.hit_definition = hit_definition

    def __eq__(self, item: object) -> bool:
        if not isinstance(item, BlastAlignment):
            return False
        if self.e_value != item.e_value:
            return False
        if self.query_interval != item.query_interval:
            return False
        if self.hit_interval != item.hit_interval:
            return False
        if self.hit_id != item.hit_id:
            return False
        if self.hit_definition != item.hit_definition:
            return False
        return super().__eq__(item)

    def __getitem__(
        self,
        index: int | slice | tuple[int | slice, int | slice],
    ) -> BlastAlignment:
        super_alignment = super().__getitem__(index)
        if super_alignment.score is None:
            raise ValueError("Cannot index a BlastAlignment without a score")
        return BlastAlignment(
            super_alignment.sequences,
            super_alignment.trace,
            super_alignment.score,
            self.e_value,
            self.query_interval,
            self.hit_interval,
            self.hit_id,
            self.hit_definition,
        )
