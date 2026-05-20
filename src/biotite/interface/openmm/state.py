from __future__ import annotations

__name__ = "biotite.interface.openmm"
__author__ = "Patrick Kunzmann"
__all__ = ["from_context", "from_state", "from_states"]

from collections.abc import Iterable
from typing import Any, cast
import numpy as np
import openmm.unit as unit
from openmm import Context, State
from biotite.structure.atoms import AtomArray, AtomArrayStack, from_template
from biotite.typing import XYZ, N, NDArray2


def from_context(template: AtomArray[N], context: Context) -> AtomArray[N]:
    """
    Parse the coordinates and box of the current state of an
    :class:`openmm.openmm.Context` into an :class:`AtomArray`.

    Parameters
    ----------
    template : AtomArray
        This structure is used as template.
        The output :class:`AtomArray` is equal to this template with the
        exception of the coordinates and the box vectors.
    context : Context
        The coordinates are parsed from the current state of this
        :class:`openmm.openmm.Context`.

    Returns
    -------
    atoms : AtomArray
        The created :class:`AtomArray`.
    """
    state = context.getState(getPositions=True)
    return from_state(template, state)


def from_state(template: AtomArray[N], state: State) -> AtomArray[N]:
    """
    Parse the coordinates and box of the given :class:`openmm.openmm.State`
    into an :class:`AtomArray`.

    Parameters
    ----------
    template : AtomArray
        This structure is used as template.
        The output :class:`AtomArray` is equal to this template with the
        exception of the coordinates and the box vectors.
    state : State
        The coordinates are parsed from this state.
        Must be created with ``getPositions=True``.

    Returns
    -------
    atoms : AtomArray
        The created :class:`AtomArray`.
    """
    coord, box = _parse_state(state)
    atoms = template.copy()
    atoms.coord = coord
    atoms.box = box
    return atoms


def from_states(
    template: AtomArray[N], states: Iterable[State]
) -> AtomArrayStack[Any, N]:
    """
    Parse the coordinates and box vectors of multiple :class:`openmm.openmm.State`
    objects into an :class:`AtomArrayStack`.

    Parameters
    ----------
    template : AtomArray
        This structure is used as template.
        The output :class:`AtomArray` is equal to this template with the
        exception of the coordinates and the box vectors.
    states : iterable of State
        The coordinates are parsed from these states.
        Must be created with ``getPositions=True``.

    Returns
    -------
    atoms : AtomArrayStack
        The created :class:`AtomArrayStack`.
    """
    coords = []
    boxes = []
    for state in states:
        coord, box = _parse_state(state)
        coords.append(coord)
        boxes.append(box)
    return from_template(template, np.stack(coords), np.stack(boxes))  # pyright: ignore[reportReturnType]


def _parse_state(
    state: State,
) -> tuple[NDArray2[N, XYZ, np.floating], NDArray2[XYZ, XYZ, np.floating]]:
    # `State.getPositions(asNumpy=True)` returns a `Quantity` wrapping a
    # numpy array, but openmm's stub describes only the unwrapped ndarray.
    coord = cast(Any, state.getPositions(asNumpy=True)).value_in_unit(unit.angstrom)
    box = cast(Any, state.getPeriodicBoxVectors(asNumpy=True)).value_in_unit(
        unit.angstrom
    )
    return coord, box
