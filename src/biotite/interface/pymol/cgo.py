__name__ = "biotite.interface.pymol"
__author__ = "Patrick Kunzmann"
__all__ = [
    "draw_cgo",
    "get_cylinder_cgo",
    "get_cone_cgo",
    "get_sphere_cgo",
    "get_point_cgo",
    "get_line_cgo",
    "get_multiline_cgo",
]

import itertools
from enum import IntEnum
import numpy as np
from biotite.interface.pymol.object import PyMOLObject
from biotite.interface.pymol.startup import get_and_set_pymol_instance

_object_counter = 0


class CGO(IntEnum):
    # List compiled from uppercase attributes in 'pymol.cgo'
    ALPHA = 25
    ALPHA_TRIANGLE = 17
    BEGIN = 2
    CHAR = 23
    COLOR = 6
    CONE = 27
    CUSTOM_CYLINDER = 15
    CYLINDER = 9
    DISABLE = 13
    DOTWIDTH = 16
    ELLIPSOID = 18
    ENABLE = 12
    END = 3
    FONT = 19
    FONT_AXES = 22
    FONT_SCALE = 20
    FONT_VERTEX = 21
    LINES = 1
    LINEWIDTH = 10
    LINE_LOOP = 2
    LINE_STRIP = 3
    NORMAL = 5
    NULL = 1
    PICK_COLOR = 31
    POINTS = 0
    QUADRIC = 26
    SAUSAGE = 14
    SPHERE = 7
    STOP = 0
    TRIANGLE = 8
    TRIANGLES = 4
    TRIANGLE_FAN = 6
    TRIANGLE_STRIP = 5
    VERTEX = 4
    WIDTHSCALE = 11


def draw_cgo(cgo_list, name=None, pymol_instance=None, delete=True):
    """
    Draw geometric shapes using *Compiled Graphics Objects* (CGOs).

    Each CGO is represented by a list of floats, which can be obtained
    via the ``get_xxx_cgo()`` functions.

    Parameters
    ----------
    cgo_list : list of list of float
        The CGOs to draw.
        It is recommended to use a ``get_xxx_cgo()`` function to obtain
        the elements for this list, if possible.
        Otherwise, shapes may be drawn incorrectly or omitted entirely,
        if a CGO is incorrectly formatted.
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
    global _object_counter
    if name is None:
        name = f"biotite_cgo_{_object_counter}"
        _object_counter += 1
    pymol_instance = get_and_set_pymol_instance(pymol_instance)
    pymol_instance.cmd.load_cgo(
        # If CGO values are integers instead of floats
        # the rendering may fail
        [float(value) for value in list(itertools.chain(*cgo_list))],
        name,
    )
    return PyMOLObject(name, pymol_instance, delete)


def get_cylinder_cgo(start, end, radius, start_color, end_color):
    """
    Get the CGO for a cylinder.

    Parameters
    ----------
    start, end : array-like, shape=(3,)
        The start and end position of the cylinder.
    radius : float
        The radius of the cylinder.
    start_color, end_color : array-like, shape=(3,)
        The color at the start and end of the cylinder given as RGB
        values in the range *(0, 1)*.

    Returns
    -------
    cgo : list of float
        The CGO representation.
    """
    _expect_length(start, "start", 3)
    _expect_length(end, "end", 3)
    _expect_length(start_color, "start_color", 3)
    _expect_length(end_color, "end_color", 3)
    _check_color(start_color)
    _check_color(end_color)
    return [CGO.CYLINDER, *start, *end, radius, *start_color, *end_color]


def get_cone_cgo(
    start, end, start_radius, end_radius, start_color, end_color, start_cap, end_cap
):
    """
    Get the CGO for a cone.

    Parameters
    ----------
    start, end : array-like, shape=(3,)
        The start and end position of the cone.
    start_radius, end_radius : float
        The radius of the cone at the start and end.
    start_color, end_color : array-like, shape=(3,)
        The color at the start and end of the cone given as RGB
        values in the range *(0, 1)*.
    start_cap, end_cap : bool
        If true, a cap is drawn at the start or end of the cone.
        Otherwise the cone is displayed as *open*.

    Returns
    -------
    cgo : list of float
        The CGO representation.
    """
    _expect_length(start, "start", 3)
    _expect_length(end, "end", 3)
    _expect_length(start_color, "start_color", 3)
    _expect_length(end_color, "end_color", 3)
    _check_color(start_color)
    _check_color(end_color)
    return [
        CGO.CONE,
        *start,
        *end,
        start_radius,
        end_radius,
        *start_color,
        *end_color,
        start_cap,
        end_cap,
    ]


def get_sphere_cgo(pos, radius, color):
    """
    Get the CGO for a sphere.

    Parameters
    ----------
    pos : array-like, shape=(3,)
        The position of the sphere.
    radius : float
        The radius of the sphere.
    color : array-like, shape=(3,)
        The color of the sphere given as RGB values in the range
        *(0, 1)*.

    Returns
    -------
    cgo : list of float
        The CGO representation.
    """
    _expect_length(pos, "pos", 3)
    _expect_length(color, "color", 3)
    _check_color(color)
    return [CGO.COLOR, *color, CGO.SPHERE, *pos, radius]


def get_point_cgo(pos, color):
    """
    Get the CGO for one or multiple points.

    Parameters
    ----------
    pos : array-like, shape=(3,), shape=(n,3)
        The position(s) of the points.
    color : array-like, shape=(3,) or shape=(n,3)
        The color of the point(s) given as RGB values in the range
        *(0, 1)*.
        Either one color can be given that is used for all points or
        an individual color for each point can be supplied.

    Returns
    -------
    cgo : list of float
        The CGO representation.
    """
    pos = np.atleast_2d(pos)
    color = _arrayfy(color, len(pos), 2)

    for p in pos:
        _expect_length(p, "pos", 3)
    for c in color:
        _expect_length(c, "color", 3)
        _check_color(c)

    vertices = []
    for p, c in zip(pos, color):
        vertices += [CGO.COLOR, *c, CGO.VERTEX, *p]

    return [CGO.BEGIN, CGO.POINTS] + vertices + [CGO.END]


def get_line_cgo(pos, color, width=1.0):
    """
    Get the CGO for a line following the given positions.

    Parameters
    ----------
    pos : array-like, shape=(n,3)
        The line follows these positions.
    color : array-like, shape=(3,) or shape=(n,3)
        The color of the line given as RGB values in the range
        *(0, 1)*.
        Either one color can be given that is used for all positions or
        an individual color for each position can be supplied.
    width : float, optional
        The rendered width of the line.
        The width is only visible after calling :func:`ray()`.

    Returns
    -------
    cgo : list of float
        The CGO representation.
    """
    color = _arrayfy(color, len(pos), 2)

    for p in pos:
        _expect_length(p, "pos", 3)
    for c in color:
        _expect_length(c, "color", 3)
        _check_color(c)

    vertices = []
    for p, c in zip(pos, color):
        vertices += [CGO.COLOR, *c, CGO.VERTEX, *p]

    return [CGO.LINEWIDTH, width, CGO.BEGIN, CGO.LINE_STRIP] + vertices + [CGO.END]


def get_multiline_cgo(start, end, color, width=1.0):
    """
    Get the CGO for one or multiple straight lines drawn from given
    start to end positions.

    Parameters
    ----------
    start, end : array-like, shape=(3,) or shape=(n,3)
        The *n* lines are drawn from the `start` to the `end` positions.
    color : array-like, shape=(3,) or shape=(n,3)
        The color of the lines given as RGB values in the range
        *(0, 1)*.
        Either one color can be given that is used for all lines or
        an individual color for each line can be supplied.
    width : float, optional
        The rendered width of the lines.
        The width is only visible after calling :func:`ray()`.

    Returns
    -------
    cgo : list of float
        The CGO representation.
    """
    start = np.atleast_2d(start)
    end = np.atleast_2d(end)
    color = _arrayfy(color, len(start), 2)

    if len(start) != len(end):
        raise IndexError(
            f"{len(start)} start positions are given, but {len(end)} end positions"
        )
    for p in start:
        _expect_length(p, "start", 3)
    for p in end:
        _expect_length(p, "end", 3)
    for c in color:
        _expect_length(c, "color", 3)
        _check_color(c)

    vertices = []
    for p1, p2, c in zip(start, end, color):
        vertices += [CGO.COLOR, *c, CGO.VERTEX, *p1, CGO.VERTEX, *p2]

    return [CGO.LINEWIDTH, width, CGO.BEGIN, CGO.LINES] + vertices + [CGO.END]


def _expect_length(values, name, length):
    if len(values) != length:
        raise IndexError(
            f"'{name}' has {len(values)} values, but {length} were expected"
        )


def _check_color(color):
    if np.any(color) < 0 or np.any(color) > 1:
        raise ValueError("Colors must be in range (0, 1)")


def _arrayfy(value, length, min_dim):
    """
    Expand value(s) to the given number of dimensions and repeat value
    `length` number of times if only a single value is given.
    """
    value = np.array(value, ndmin=min_dim)
    if len(value) == 1 and length > 1:
        value = np.repeat(value, length, axis=0)
    elif len(value) != length:
        raise IndexError(f"Expected {length} values, but got {len(value)}")
    return value
