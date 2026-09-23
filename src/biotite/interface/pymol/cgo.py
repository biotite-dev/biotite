from __future__ import annotations

__name__ = "biotite.interface.pymol"
__author__ = "Patrick Kunzmann"
__all__ = [
    "CGO",
    "draw_cgo",
    "get_cylinder_cgo",
    "get_cone_cgo",
    "get_sphere_cgo",
    "get_point_cgo",
    "get_line_cgo",
    "get_multiline_cgo",
]

from collections.abc import Sequence
from enum import IntEnum
from typing import Any, TypeAlias
import numpy as np
from numpy.typing import ArrayLike
from biotite.interface.pymol.object import PyMOLObject
from biotite.interface.pymol.startup import PyMOLInstance, get_and_set_pymol_instance
from biotite.typing import NDArray1

# Iterable-and-len vector input; unlike numpy's ArrayLike this excludes scalars
# so it can be unpacked with `*` to splat the elements into a CGO command list.
_Vector: TypeAlias = Sequence[float] | NDArray1[Any, np.floating]

_object_counter: int = 0


class CGO:
    """
    A *Compiled Graphics Object* (CGO) describing a geometric shape that
    can be drawn in *PyMOL*.

    Under the hood, a CGO is a sequence of floating point values
    comprising directives (see :class:`CGO.Type`) and their arguments.
    Usually, a :class:`CGO` is created via one of the ``get_xxx_cgo()``
    functions and drawn via :func:`draw_cgo()`.

    Parameters
    ----------
    values : array-like, shape=(k,), dtype=float
        The raw sequence of CGO directives and their arguments.
        Integer values are converted to floats, as *PyMOL* may
        otherwise fail to render the shape.

    Attributes
    ----------
    values : ndarray, shape=(k,), dtype=float
        The raw values of the CGO.

    Examples
    --------

    >>> cgo = get_sphere_cgo(pos=(1.0, 2.0, 3.0), radius=1.5, color=(1.0, 0.0, 0.0))
    >>> print(cgo)
    CGO([6.0, 1.0, 0.0, 0.0, 7.0, 1.0, 2.0, 3.0, 1.5])
    >>> print(cgo.values[0] == CGO.Type.COLOR)
    True
    """

    class Type(IntEnum):
        """
        The directives a CGO may contain.
        """

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

    def __init__(self, values: ArrayLike | Sequence[float | np.floating]) -> None:
        # If CGO values are integers instead of floats
        # the rendering may fail
        values = np.asarray(values, dtype=float)
        if values.ndim != 1:
            raise ValueError("CGO values must be one-dimensional")
        self._values: NDArray1[Any, np.floating] = values

    @property
    def values(self) -> NDArray1[Any, np.floating]:
        return self._values

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CGO):
            return False
        return np.array_equal(self._values, other._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"CGO({self._values.tolist()})"


def draw_cgo(
    cgos: Sequence[CGO],
    name: str | None = None,
    pymol_instance: PyMOLInstance | None = None,
    delete: bool = True,
) -> PyMOLObject:
    """
    Draw geometric shapes using *Compiled Graphics Objects* (CGOs).

    Parameters
    ----------
    cgos : list of CGO
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
    for cgo in cgos:
        if not isinstance(cgo, CGO):
            raise TypeError(f"Expected 'CGO', but got '{type(cgo).__name__}'")
    pymol_instance = get_and_set_pymol_instance(pymol_instance)
    values = np.concatenate([cgo.values for cgo in cgos]) if cgos else np.zeros(0)
    pymol_instance.cmd.load_cgo(values.tolist(), name)
    return PyMOLObject(name, pymol_instance, delete)


def get_cylinder_cgo(
    start: _Vector,
    end: _Vector,
    radius: float,
    start_color: _Vector,
    end_color: _Vector,
) -> CGO:
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
    cgo : CGO
        The CGO representation.
    """
    _expect_length(start, "start", 3)
    _expect_length(end, "end", 3)
    _expect_length(start_color, "start_color", 3)
    _expect_length(end_color, "end_color", 3)
    _check_color(start_color)
    _check_color(end_color)
    return CGO([CGO.Type.CYLINDER, *start, *end, radius, *start_color, *end_color])


def get_cone_cgo(
    start: _Vector,
    end: _Vector,
    start_radius: float,
    end_radius: float,
    start_color: _Vector,
    end_color: _Vector,
    start_cap: bool,
    end_cap: bool,
) -> CGO:
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
    cgo : CGO
        The CGO representation.
    """
    _expect_length(start, "start", 3)
    _expect_length(end, "end", 3)
    _expect_length(start_color, "start_color", 3)
    _expect_length(end_color, "end_color", 3)
    _check_color(start_color)
    _check_color(end_color)
    return CGO(
        [
            CGO.Type.CONE,
            *start,
            *end,
            start_radius,
            end_radius,
            *start_color,
            *end_color,
            start_cap,
            end_cap,
        ]
    )


def get_sphere_cgo(pos: _Vector, radius: float, color: _Vector) -> CGO:
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
    cgo : CGO
        The CGO representation.
    """
    _expect_length(pos, "pos", 3)
    _expect_length(color, "color", 3)
    _check_color(color)
    return CGO([CGO.Type.COLOR, *color, CGO.Type.SPHERE, *pos, radius])


def get_point_cgo(pos: ArrayLike, color: ArrayLike) -> CGO:
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
    cgo : CGO
        The CGO representation.
    """
    pos_array = np.atleast_2d(pos)
    color_array = _arrayfy(color, len(pos_array), 2)

    for p in pos_array:
        _expect_length(p, "pos", 3)
    for c in color_array:
        _expect_length(c, "color", 3)
        _check_color(c)

    vertices: list[float] = []
    for p, c in zip(pos_array, color_array):
        vertices += [CGO.Type.COLOR, *c, CGO.Type.VERTEX, *p]

    return CGO([CGO.Type.BEGIN, CGO.Type.POINTS, *vertices, CGO.Type.END])


def get_line_cgo(pos: ArrayLike, color: ArrayLike, width: float = 1.0) -> CGO:
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
    cgo : CGO
        The CGO representation.
    """
    pos_array = np.atleast_2d(pos)
    color_array = _arrayfy(color, len(pos_array), 2)

    for p in pos_array:
        _expect_length(p, "pos", 3)
    for c in color_array:
        _expect_length(c, "color", 3)
        _check_color(c)

    vertices: list[Any] = []
    for p, c in zip(pos_array, color_array):
        vertices += [CGO.Type.COLOR, *c, CGO.Type.VERTEX, *p]

    return CGO(
        [
            CGO.Type.LINEWIDTH,
            width,
            CGO.Type.BEGIN,
            CGO.Type.LINE_STRIP,
            *vertices,
            CGO.Type.END,
        ]
    )


def get_multiline_cgo(
    start: ArrayLike, end: ArrayLike, color: ArrayLike, width: float = 1.0
) -> CGO:
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
    cgo : CGO
        The CGO representation.
    """
    start_array = np.atleast_2d(start)
    end_array = np.atleast_2d(end)
    color_array = _arrayfy(color, len(start_array), 2)

    if len(start_array) != len(end_array):
        raise IndexError(
            f"{len(start_array)} start positions are given, "
            f"but {len(end_array)} end positions"
        )
    for p in start_array:
        _expect_length(p, "start", 3)
    for p in end_array:
        _expect_length(p, "end", 3)
    for c in color_array:
        _expect_length(c, "color", 3)
        _check_color(c)

    vertices: list[Any] = []
    for p1, p2, c in zip(start_array, end_array, color_array):
        vertices += [CGO.Type.COLOR, *c, CGO.Type.VERTEX, *p1, CGO.Type.VERTEX, *p2]

    return CGO(
        [
            CGO.Type.LINEWIDTH,
            width,
            CGO.Type.BEGIN,
            CGO.Type.LINES,
            *vertices,
            CGO.Type.END,
        ]
    )


def _expect_length(values: Any, name: str, length: int) -> None:
    if len(values) != length:
        raise IndexError(
            f"'{name}' has {len(values)} values, but {length} were expected"
        )


def _check_color(color: _Vector) -> None:
    if np.any(color) < 0 or np.any(color) > 1:
        raise ValueError("Colors must be in range (0, 1)")


def _arrayfy(value: ArrayLike, length: int, min_dim: int) -> np.ndarray:
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
