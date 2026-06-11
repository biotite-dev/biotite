# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application.viennarna"
__author__ = "Patrick Kunzmann"
__all__ = ["build_constraint_string"]

import numpy as np
from biotite.structure.pseudoknots import pseudoknots
from biotite.typing import C2, N, NDArray1, NDArray2


def build_constraint_string(
    sequence_length: int,
    pairs: NDArray2[N, C2, np.integer] | None = None,
    paired: NDArray1[N, np.integer | np.bool_] | None = None,
    unpaired: NDArray1[N, np.integer | np.bool_] | None = None,
    downstream: NDArray1[N, np.integer | np.bool_] | None = None,
    upstream: NDArray1[N, np.integer | np.bool_] | None = None,
) -> str:
    """
    Build a ViennaRNA constraint string.

    Parameters
    ----------
    sequence_length : int
        The length of the string to be built.
    pairs : ndarray, shape=(n,2), dtype=int, optional
        Positions of constrained base pairs.
    paired : ndarray, shape=(n,), dtype=int or dtype=bool, optional
        Positions of bases that are paired with any other base.
    unpaired : ndarray, shape=(n,), dtype=int or dtype=bool, optional
        Positions of bases that are unpaired.
    downstream : ndarray, shape=(n,), dtype=int or dtype=bool, optional
        Positions of bases that are paired with any downstream base.
    upstream : ndarray, shape=(n,), dtype=int or dtype=bool, optional
        Positions of bases that are paired with any upstream base.

    Returns
    -------
    constraints : str
        The constraint string.
    """
    constraints = np.full(sequence_length, ".", dtype="U1")

    if pairs is not None:
        pairs = np.asarray(pairs)
        # Ensure that pairs do not contain pseudoknots
        if (pseudoknots(pairs, max_pseudoknot_order=1) == -1).any():
            raise ValueError("Given pairs include pseudoknots")
        # Ensure the lower base comes first for each pair
        pairs = np.sort(pairs, axis=-1)
        _set_constraints(constraints, pairs[:, 0], "(")
        _set_constraints(constraints, pairs[:, 1], ")")

    _set_constraints(constraints, paired, "|")
    _set_constraints(constraints, unpaired, "x")
    _set_constraints(constraints, downstream, "<")
    _set_constraints(constraints, upstream, ">")

    return "".join(constraints)


def _set_constraints(
    constraints: NDArray1[N, np.str_],
    index: NDArray1[N, np.integer | np.bool_] | None,
    character: str,
) -> None:
    if index is None:
        return

    # Search for conflicts with other constraints
    potential_conflict_indices = np.where(constraints[index] != ".")[0]
    if len(potential_conflict_indices) > 0:
        conflict_i = index[potential_conflict_indices[0]]
        raise ValueError(
            f"Constraint '{character}' at position {conflict_i} "
            f"conflicts with existing constraint '{constraints[conflict_i]}'"
        )

    constraints[index] = character
