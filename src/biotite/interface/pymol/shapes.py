__name__ = "biotite.interface.pymol"
__author__ = "Patrick Kunzmann"
__all__ = ["draw_arrows", "draw_box"]

import numpy as np
from biotite.interface.pymol.cgo import (
    _arrayfy,
    draw_cgo,
    get_cone_cgo,
    get_cylinder_cgo,
    get_multiline_cgo,
)


def draw_arrows(
    start,
    end,
    radius=0.1,
    head_radius=0.20,
    head_length=0.5,
    color=(0.5, 0.5, 0.5),
    head_color=None,
    name=None,
    pymol_instance=None,
    delete=True,
):
    """
    Draw three-dimensional arrows using *Compiled Graphics Objects* (CGOs).

    Parameters
    ----------
    start, end : array-like, shape=(n,3)
        The start and end position of each arrow.
    radius, head_radius : float or array-like, shape=(n,), optional
        The radius of the tail and head for each arrow.
        Uniform for all arrows, if a single value is given.
    head_length : float or array-like, shape=(n,), optional
        The length of each arrow head.
        Uniform for all arrows, if a single value is given.
    color, head_color : array-like, shape=(3,) or shape=(n,3), optional
        The color of the tail and head for each arrow, given as RGB
        values in the range *(0, 1)*.
        Uniform for all arrows, if a single value is given.
        If no `head_color` is given, the arrows are single-colored.
    name : str, optional
        The name of the newly created CGO object.
        If omitted, a unique name is generated.
    pymol_instance : module or SingletonPyMOL or PyMOL, optional
        If *PyMOL* is used in library mode, the :class:`PyMOL`
        or :class:`SingletonPyMOL` object is given here.
        If otherwise *PyMOL* is used in GUI mode, the :mod:`pymol`
        module is given.
        By default the currently active *PyMOL* instance is used.
        If no *PyMOL* instance is currently running,
        *PyMOL* is started in library mode.
    delete : bool, optional
        If set to true, the underlying *PyMOL* object will be removed from the *PyMOL*
        session, when the returned :class:`PyMOLObject` is garbage collected.

    Returns
    -------
    pymol_object : PyMOLObject
        The created :class:`PyMOLObject` representing the drawn CGOs.
    """
    if head_color is None:
        head_color = color

    start = np.asarray(start)
    end = np.asarray(end)
    if start.ndim != 2 or end.ndim != 2:
        raise IndexError("Expected 2D array for start and end positions")
    if len(start) != len(end):
        raise IndexError(
            f"Got {len(start)} start positions, but expected {len(end)} end positions"
        )
    expected_length = len(start)
    radius = _arrayfy(radius, expected_length, 1)
    head_radius = _arrayfy(head_radius, expected_length, 1)
    head_length = _arrayfy(head_length, expected_length, 1)
    color = _arrayfy(color, expected_length, 2)
    head_color = _arrayfy(head_color, expected_length, 2)

    normal = (end - start) / np.linalg.norm(end - start, axis=-1)[:, np.newaxis]
    middle = end - normal * head_length[:, np.newaxis]

    cgo_list = []
    for i in range(len(start)):
        cgo_list.extend(
            [
                get_cylinder_cgo(start[i], middle[i], radius[i], color[i], color[i]),
                get_cone_cgo(
                    middle[i],
                    end[i],
                    head_radius[i],
                    0.0,
                    head_color[i],
                    head_color[i],
                    True,
                    False,
                ),
            ]
        )

    return draw_cgo(cgo_list, name, pymol_instance, delete)


def draw_box(
    box,
    color=(0, 1, 0),
    width=1.0,
    origin=None,
    name=None,
    pymol_instance=None,
    delete=True,
):
    """
    Draw a box using *Compiled Graphics Objects* (CGOs).

    This can be used to draw the unit cell or periodic box of an
    :class:`AtomArray`.

    Parameters
    ----------
    box : array-like, shape=(3,3)
        The three box vectors.
    color : array-like, shape=(3,), optional
        The color of the box, given as RGB
        values in the range *(0, 1)*.
    width : float, optional
        The width of the drawn lines.
    origin : array-like, shape=(3,), optional
        If given the box origin is drawn at the given position instead of the
        coordinate origin.
    name : str, optional
        The name of the newly created CGO object.
        If omitted, a unique name is generated.
    pymol_instance : module or SingletonPyMOL or PyMOL, optional
        If *PyMOL* is used in library mode, the :class:`PyMOL`
        or :class:`SingletonPyMOL` object is given here.
        If otherwise *PyMOL* is used in GUI mode, the :mod:`pymol`
        module is given.
        By default the currently active *PyMOL* instance is used.
        If no *PyMOL* instance is currently running,
        *PyMOL* is started in library mode.
    delete : bool, optional
        If set to true, the underlying *PyMOL* object will be removed from the *PyMOL*
        session, when the returned :class:`PyMOLObject` is garbage collected.

    Returns
    -------
    pymol_object : PyMOLObject
        The created :class:`PyMOLObject` representing the drawn CGOs.
    """
    box = np.asarray(box)
    if origin is None:
        origin = np.zeros(3)
    else:
        origin = np.asarray(origin)

    starts = []
    ends = []
    for direction_dim in (0, 1, 2):
        plane_dim1, plane_dim2 = [dim for dim in (0, 1, 2) if dim != direction_dim]
        starts.append(origin)
        ends.append(origin + box[direction_dim])

        starts.append(origin + box[plane_dim1])
        ends.append(origin + box[plane_dim1] + box[direction_dim])

        starts.append(origin + box[plane_dim2])
        ends.append(origin + box[plane_dim2] + box[direction_dim])

        starts.append(origin + box[plane_dim1] + box[plane_dim2])
        ends.append(origin + box[plane_dim1] + box[plane_dim2] + box[direction_dim])

    return draw_cgo(
        [get_multiline_cgo(starts, ends, color, width)], name, pymol_instance, delete
    )
