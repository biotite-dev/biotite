__name__ = "biotite.interface.pymol"
__author__ = "Patrick Kunzmann"
__all__ = ["PyMOLObject", "NonexistentObjectError", "ModifiedObjectError"]

import numbers
from enum import Enum
from functools import wraps
import numpy as np
import biotite.structure as struc
from biotite.interface.pymol.convert import (
    from_model,
    to_model,
)
from biotite.interface.pymol.startup import get_and_set_pymol_instance


def validate(method):
    """
    Check if the object name still exists and if the atom count has
    been modified.
    If this is the case, raise the appropriate exception.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self._check_existence()
        if self._type == PyMOLObject.Type.MOLECULE:
            new_atom_count = self._cmd.count_atoms(f"model {self._name}")
            if new_atom_count != self._atom_count:
                raise ModifiedObjectError(
                    f"The number of atoms in the object changed "
                    f"from the original {self._atom_count} atoms "
                    f" to {new_atom_count} atoms"
                )
        return method(self, *args, **kwargs)

    return wrapper


class PyMOLObject:
    """
    A handle to a *PyMOL object* (*PyMOL model*), usually created by the static
    :meth:`from_structure()` method.

    This class conveniently provides thin wrappers around *PyMOL* commands that
    allows *NumPy*-style atom indices to select atoms instead of *PyMOL* selection
    expressions.
    The following *PyMOL* commands are supported:

    - :meth:`alter()`
    - :meth:`cartoon()`
    - :meth:`center()`
    - :meth:`clip()`
    - :meth:`color()`
    - :meth:`desaturate()`
    - :meth:`disable()`
    - :meth:`distance()`
    - :meth:`dss()`
    - :meth:`enable()`
    - :meth:`hide()`
    - :meth:`indicate()`
    - :meth:`label()`
    - :meth:`orient()`
    - :meth:`origin()`
    - :meth:`select()`
    - :meth:`set()`
    - :meth:`set_bond()`
    - :meth:`show()`
    - :meth:`show_as()`
    - :meth:`smooth()`
    - :meth:`unset()`
    - :meth:`unset_bond()`
    - :meth:`zoom()`

    Instances of this class become invalid, when atoms are added to or
    are deleted from the underlying *PyMOL* object.
    Calling methods of such an an invalidated object raises an
    :exc:`ModifiedObjectError`.
    Likewise, calling methods of an object, of which the underlying
    *PyMOL* object does not exist anymore, raises an
    :exc:`NonexistentObjectError`.

    Parameters
    ----------
    name : str
        The name of the *PyMOL* object.
    pymol_instance : module or SingletonPyMOL or PyMOL, optional
        If *PyMOL* is used in library mode, the :class:`PyMOL` or
        :class:`SingletonPyMOL` object is given here.
        If otherwise *PyMOL* is used in GUI mode, the :mod:`pymol` module is given.
        By default the currently active *PyMOL* instance is used.
        If no *PyMOL* instance is currently running, *PyMOL* is started in library mode.
    delete : PyMOL, optional
        If set to true, the underlying *PyMOL* object will be removed
        from the *PyMOL* session, when this object is garbage collected.

    Attributes
    ----------
    name : str
        The name of the *PyMOL* object.
    object_type : PyMOLObject.Type
        The type of this *PyMOL* object.
    """

    class Type(Enum):
        """
        Defines what this :class:`PyMOLObject` represents.
        """

        MOLECULE = "object:molecule"
        MAP = "object:map"
        MESH = "object:mesh"
        SLICE = "object:slice"
        SURFACE = "object:surface"
        MEASUREMENT = "object:measurement"
        CGO = "object:cgo"
        GROUP = "object:group"
        VOLUME = "object:volume"
        SELECTION = "selection"

    _object_counter = 0
    _color_counter = 0

    def __init__(self, name, pymol_instance=None, delete=True):
        self._name = name
        self._pymol = get_and_set_pymol_instance(pymol_instance)
        self._delete = delete
        self._cmd = self._pymol.cmd
        self._type = PyMOLObject.Type(self._cmd.get_type(self._name))
        self._check_existence()
        if self._type == PyMOLObject.Type.MOLECULE:
            self._atom_count = self._cmd.count_atoms(f"%{self._name}")

    def __del__(self):
        if self._delete:
            try:
                # Try to delete this object from PyMOL
                # Fails if PyMOL itself is already garbage collected
                self._cmd.delete(self._name)
            except Exception:
                pass

    @staticmethod
    def from_structure(
        atoms, name=None, pymol_instance=None, delete=True, delocalize_bonds=False
    ):
        """
        Create a :class:`PyMOLObject` from an :class:`AtomArray` or
        :class:`AtomArrayStack` and add it to the *PyMOL* session.

        Parameters
        ----------
        atoms : AtomArray or AtomArrayStack
            The structure to be converted.
        name : str, optional
            The name of the newly created *PyMOL* object.
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
            If set to true, the underlying *PyMOL* object will be
            removed from the *PyMOL* session, when this object is
            garbage collected.
        delocalize_bonds : bool, optional
            If set to true, use *PyMOL*'s delocalized bond order for aromatic bonds.
            Otherwise, always use formal bond orders.

        Returns
        -------
        pymol_object : PyMOLObject
            The :class:`PyMOLObject` representing the given structure.
        """
        pymol_instance = get_and_set_pymol_instance(pymol_instance)
        cmd = pymol_instance.cmd

        if name is None:
            name = f"biotite_{PyMOLObject._object_counter}"
            PyMOLObject._object_counter += 1

        if isinstance(atoms, struc.AtomArray) or (
            isinstance(atoms, struc.AtomArrayStack) and atoms.stack_depth == 1
        ):
            model = to_model(atoms, delocalize_bonds)
            cmd.load_model(model, name)
        elif isinstance(atoms, struc.AtomArrayStack):
            # Use first model as template
            model = to_model(atoms[0])
            cmd.load_model(model, name)
            # Append states corresponding to all following models
            for coord in atoms.coord[1:]:
                cmd.load_coordset(coord, name)
        else:
            raise TypeError("Expected 'AtomArray' or 'AtomArrayStack'")

        return PyMOLObject(name, pymol_instance, delete)

    def to_structure(self, state=None, altloc="first", include_bonds=False):
        """
        Convert this object into an :class:`AtomArray` or
        :class:`AtomArrayStack`.

        The returned :class:`AtomArray` contains the optional annotation
        categories ``b_factor``, ``occupancy`` and ``charge``.

        Parameters
        ----------
        state : int, optional
            If this parameter is given, the function will return an
            :class:`AtomArray` corresponding to the given state of the
            *PyMOL* object.
            If this parameter is omitted, an :class:`AtomArrayStack`
            containing all states will be returned, even if the *PyMOL*
            object contains only one state.
        altloc : {'first', 'occupancy', 'all'}
            This parameter defines how *altloc* IDs are handled:

            - ``'first'`` - Use atoms that have the first
              *altloc* ID appearing in a residue.
            - ``'occupancy'`` - Use atoms that have the *altloc* ID
              with the highest occupancy for a residue.
            - ``'all'`` - Use all atoms.
              Note that this leads to duplicate atoms.
              When this option is chosen, the ``altloc_id``
              annotation array is added to the returned structure.

        include_bonds : bool, optional
            If set to true, an associated :class:`BondList` will be created
            for the returned structure.

        Returns
        -------
        structure : AtomArray or AtomArrayStack
            The converted structure.
            Whether an :class:`AtomArray` or :class:`AtomArrayStack` is
            returned depends on the `state` parameter.
        """
        if state is None:
            model = self._cmd.get_model(self._name, state=1)
            template = from_model(model, include_bonds)
            expected_length = None
            coord = []
            for i in range(self._cmd.count_states(self._name)):
                state_coord = self._cmd.get_coordset(self._name, state=i + 1)
                if expected_length is None:
                    expected_length = len(state_coord)
                elif len(state_coord) != expected_length:
                    raise ValueError("The models have different numbers of atoms")
                coord.append(state_coord)
            coord = np.stack(coord)
            structure = struc.from_template(template, coord)

        else:
            model = self._cmd.get_model(self._name, state=state)
            structure = from_model(model, include_bonds)

        # Filter altloc IDs and return
        if altloc == "occupancy":
            structure = structure[
                ...,
                struc.filter_highest_occupancy_altloc(
                    structure, structure.altloc_id, structure.occupancy
                ),
            ]
            structure.del_annotation("altloc_id")
            return structure
        elif altloc == "first":
            structure = structure[
                ..., struc.filter_first_altloc(structure, structure.altloc_id)
            ]
            structure.del_annotation("altloc_id")
            return structure
        elif altloc == "all":
            return structure
        else:
            raise ValueError(f"'{altloc}' is not a valid 'altloc' option")

    @property
    def name(self):
        return self._name

    @property
    def object_type(self):
        return self._type

    def exists(self):
        """
        Check whether the underlying *PyMOL* object still exists.

        Returns
        -------
        bool
            True if the *PyMOL* session contains an object with the name
            of this :class:`PyMOLObject`, false otherwise.
        """
        return self._name in self._cmd.get_names("all", enabled_only=False)

    def _check_existence(self):
        if not self.exists():
            raise NonexistentObjectError(
                f"A PyMOL object with the name {self._name} does not exist anymore"
            )

    @validate
    def where(self, index):
        """
        Convert a *Biotite*-compatible atom selection index
        (integer, slice, boolean mask, index array) into a *PyMOL*
        selection expression.

        Parameters
        ----------
        index : int or slice or ndarray, dtype=bool or ndarray, dtype=int
            The boolean mask to be converted into a selection string.

        Returns
        -------
        expression : str
            A *PyMOL* compatible selection expression.
        """
        if self._type != PyMOLObject.Type.MOLECULE:
            raise TypeError(
                "Can only create atom selection strings "
                "for PyMOL objects representing molecules"
            )

        if isinstance(index, numbers.Integral):
            # PyMOLs indexing starts at 1
            return f"%{self._name} and index {index}"

        elif isinstance(index, np.ndarray) and index.dtype == bool:
            mask = index
            if len(mask) != self._atom_count:
                raise IndexError(
                    f"Mask has length {len(mask)}, but the number of "
                    f"atoms in the PyMOL model is {self._atom_count}"
                )

        else:
            # Convert any other index type into a boolean mask
            mask = np.zeros(self._atom_count, dtype=bool)
            mask[index] = True

        # Indices where the mask changes from True to False
        # or from False to True
        # The '+1' makes each index refer to the position
        # after the change i.e. the new value
        changes = np.where(np.diff(mask))[0] + 1
        # If first element is True, insert index 0 at start
        # -> the first change is always from False to True
        if mask[0]:
            changes = np.concatenate(([0], changes))
        # If the last element is True, insert append length of mask
        # as exclusive stop index
        # -> the last change is always from True to False
        if mask[-1]:
            changes = np.concatenate((changes, [len(mask)]))
        # -> Changes are alternating (F->T, T->F, F->T, ..., F->T, T->F)
        # Reshape into pairs ([F->T, T->F], [F->T, T->F], ...)
        # -> these are the intervals where the mask is True
        intervals = np.reshape(changes, (-1, 2))

        if len(intervals) > 0:
            # Convert interval into selection string
            # Two things to note:
            # - PyMOLs indexing starts at 1-> 'start+1'
            # - Stop in 'intervals' is exclusive -> 'stop+1-1' -> 'stop'
            index_selection = " or ".join(
                [f"index {start + 1}-{stop}" for start, stop in intervals]
            )
            # Constrain the selection to given object name
            return f"%{self._name} and ({index_selection})"
        else:
            return "none"

    def _into_selection(self, selection, not_none=False):
        """
        Turn a boolean mask into a *PyMOL* selection expression or
        restrict an selection expression to the current *PyMOL* object.
        """
        if selection is None:
            return f"%{self._name}"
        elif isinstance(selection, str):
            return f"%{self._name} and ({selection})"
        else:
            sel = self.where(np.asarray(selection))
            if sel == "none" and not_none:
                raise ValueError("Selection contains no atoms")
            return sel

    @validate
    def alter(self, selection, expression):
        """
        Change atomic properties using an expression evaluated
        within a temporary namespace for each atom.

        This method is a thin wrapper around the *PyMOL* ``alter()``
        command.

        Parameters
        ----------
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
        expression : str
            The properties of the selected atoms are changed based on
            this expression.
        """
        self._cmd.alter(self._into_selection(selection), expression)

    @validate
    def cartoon(self, type, selection=None):
        """
        Change the default cartoon representation for a selection
        of atoms.

        This method is a thin wrapper around the *PyMOL* ``cartoon()``
        command.

        Parameters
        ----------
        type : str
            One of

            - ``'automatic'``,
            - ``'skip'``,
            - ``'loop'``,
            - ``'rectangle'``,
            - ``'oval'``,
            - ``'tube'``,
            - ``'arrow'`` or
            - ``'dumbbell'``.

        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        """
        self._cmd.cartoon(type, self._into_selection(selection))

    @validate
    def center(self, selection=None, state=None, origin=True):
        """
        Translate the window, the clipping slab, and the
        origin to a point centered within the atom selection.

        This method is a thin wrapper around the *PyMOL* ``center()``
        command.

        Parameters
        ----------
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        state : int, optional
            The state to apply the command on.
            By default, the command is applied on all states of this
            *PyMOL* object.
        origin : bool, optional
            If set to false, the origin is left unchanged.
        """
        state = 0 if state is None else state
        self._cmd.center(self._into_selection(selection), state, int(origin))

    @validate
    def clip(self, mode, distance, selection=None, state=None):
        """
        Alter the positions of the near and far clipping planes.

        This method is a thin wrapper around the *PyMOL* ``clip()``
        command.

        Parameters
        ----------
        mode : {'near', 'far', 'move', 'slab', 'atoms'}

            - ``near`` - Move the near plane
            - ``far`` - Move the far plane
            - ``move`` - Move slab
            - ``slab`` - Set slab thickness
            - ``atoms`` - clip selected atoms with the given buffer

        distance : float
            The meaning of this parameter depends on `mode`.
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        state : int, optional
            The state to apply the command on.
            By default, the command is applied on all states of this
            *PyMOL* object.
        """
        state = 0 if state is None else state
        self._cmd.clip(mode, distance, self._into_selection(selection), state)

    @validate
    def color(self, color, selection=None, representation=None):
        """
        Change the color of atoms.

        This method is a thin wrapper around the *PyMOL* ``color()``
        or ``set("xxx_color")`` command.

        Parameters
        ----------
        color : str or tuple(float, float, float)
            Either a *PyMOL* color name or a tuple containing an RGB
            value (0.0 to 1.0).
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        representation : {"sphere", "surface", "mesh", "dot", "cartoon", "ribbon"}, optional
            Colors only the given representation by internally calling
            ``set("xxx_color")``.
            By default, all representations are affected, i.e. ``color()``
            is called internally.

        Notes
        -----
        If an RGB color is given, the color is registered as a unique
        named color via the ``set_color()`` command.
        """
        if not isinstance(color, str):
            color_name = f"biotite_color_{PyMOLObject._color_counter}"
            PyMOLObject._color_counter += 1
            self._cmd.set_color(color_name, tuple(color))
        else:
            color_name = color
            registered = [name for name, _ in self._cmd.get_color_indices()]
            if color_name not in registered:
                raise ValueError(f"Unknown color '{color_name}'")

        if representation is None:
            self._cmd.color(color_name, self._into_selection(selection))
        else:
            if representation not in (
                "sphere",
                "surface",
                "mesh",
                "dot",
                "cartoon",
                "ribbon",
            ):
                raise ValueError(
                    f"'{representation}' is not a supported representation"
                )
            self._cmd.set(
                f"{representation}_color", color_name, self._into_selection(selection)
            )

    @validate
    def surface_color(self, color, selection=None):
        """
        Change the color of displayed surfaces.

        This method is a thin wrapper around *PyMOL*
        ``set("surface_color")``.

        Parameters
        ----------
        color : str or tuple(float, float, float)
            Either a *PyMOL* color name or a tuple containing an RGB
            value (0.0 to 1.0).
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.

        Notes
        -----
        If an RGB color is given, the color is registered as a unique
        named color via the ``set_color()`` command.
        """
        color_name = self._created_named_color(color)
        self._cmd.set("surface_color", color_name, self._into_selection(selection))

    @validate
    def desaturate(self, selection=None, a=0.5):
        """
        Desaturate the colors of the selected atoms.

        This method is a thin wrapper around the *PyMOL*
        ``desaturate()`` command.

        Parameters
        ----------
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        a : float
            A desaturation factor between 0.0 and 1.0.
        """
        self._cmd.desaturate(self._into_selection(selection), a)

    @validate
    def disable(self, selection=None):
        """
        Turn off display of the selected atoms.

        This method is a thin wrapper around the *PyMOL* ``disable()``
        command.

        Parameters
        ----------
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        """
        self._cmd.disable(self._into_selection(selection))

    @validate
    def distance(
        self,
        name,
        selection1,
        selection2,
        cutoff=None,
        mode=None,
        show_label=True,
        width=None,
        length=None,
        gap=None,
    ):
        """
        Create a new distance object between two atom selections.

        This method is a thin wrapper around the *PyMOL* ``distance()``
        command.

        Parameters
        ----------
        name : str
            Name of the distance object to create.
        selection1, selection2 : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
        cutoff : float, optional
            The longest distance to show.
        mode : {0, 1, 2, 3, 4}, optional

            - ``0`` - All interatomic distances
            - ``1`` - Only bond distances
            - ``2`` - Only polar contact distances
            - ``3`` - All interatomic distances,
              use distance_exclusion setting
            - ``4`` - Distance between centroids

        show_label : bool, optional
            Whether to show the distance as label.
        width, length, gap : float optional
            The width and length of each dash and the gap length between
            the dashes.
        """
        self._cmd.distance(
            name,
            self._into_selection(selection1),
            self._into_selection(selection2),
            cutoff,
            mode,
            label=int(show_label),
            width=width,
            length=length,
            gap=gap,
        )

    @validate
    def dss(self, selection=None, state=None):
        """
        Determine the secondary structure of the selected atoms.

        This method is a thin wrapper around the *PyMOL* ``dss()``
        command.

        Parameters
        ----------
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        state : int, optional
            The state to apply the command on.
            By default, the command is applied on all states of this
            *PyMOL* object.
        """
        state = 0 if state is None else state
        self._cmd.dss(self._into_selection(selection), state)

    @validate
    def enable(self, selection=None):
        """
        Turn on display of the selected atoms.

        This method is a thin wrapper around the *PyMOL* ``enable()``
        command.

        Parameters
        ----------
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        """
        self._cmd.enable(self._into_selection(selection))

    @validate
    def hide(self, representation, selection=None):
        """
        Turn off an atom representation (e.g. sticks, spheres, etc.).

        This method is a thin wrapper around the *PyMOL* ``hide()``
        command.

        Parameters
        ----------
        representation : str
            One of

            - ``'lines'``,
            - ``'spheres'``,
            - ``'mesh'``,
            - ``'ribbon'``,
            - ``'cartoon'``,
            - ``'sticks'``,
            - ``'dots'``,
            - ``'surface'``,
            - ``'label'``,
            - ``'extent'``,
            - ``'nonbonded'``,
            - ``'nb_spheres'``,
            - ``'slice'`` or
            - ``'cell'``.

        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        """
        self._cmd.hide(representation, self._into_selection(selection))

    @validate
    def indicate(self, selection=None):
        """
        Show a visual representation of the selected atoms.

        This method is a thin wrapper around the *PyMOL* ``indicate()``
        command.

        Parameters
        ----------
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        """
        self._cmd.indicate(self._into_selection(selection))

    @validate
    def label(self, selection, text):
        """
        Label the selected atoms.

        This method is a thin wrapper around the *PyMOL* ``label()``
        command.

        Parameters
        ----------
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
        text : str
            The label text.
        """
        self._cmd.label(self._into_selection(selection), f'"{text}"')

    @validate
    def orient(self, selection=None, state=None):
        """
        Align the principal components of the selected atoms with the
        *xyz* axes.

        This method is a thin wrapper around the *PyMOL* ``orient()``
        command.

        Parameters
        ----------
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        state : int, optional
            The state to apply the command on.
            By default, the command is applied on all states of this
            *PyMOL* object.
        """
        state = 0 if state is None else state
        self._cmd.orient(self._into_selection(selection, True), state)

    @validate
    def origin(self, selection=None, state=None):
        """
        Set the center of rotation about the selected atoms.

        This method is a thin wrapper around the *PyMOL* ``origin()``
        command.

        Parameters
        ----------
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        state : int, optional
            The state to apply the command on.
            By default, the command is applied on all states of this
            *PyMOL* object.
        """
        state = 0 if state is None else state
        self._cmd.origin(selection=self._into_selection(selection), state=state)

    @validate
    def select(self, name, selection=None):
        """
        Create a named selection object from the selected atoms.

        This method is a thin wrapper around the *PyMOL* ``select()``
        command.

        Parameters
        ----------
        name : str
            Name of the selection object to create.
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        """
        self._cmd.select(name, self._into_selection(selection))

    @validate
    def set(self, name, value, selection=None, state=None):
        """
        Change per-atom settings.

        This method is a thin wrapper around the *PyMOL* ``set()``
        command.

        Parameters
        ----------
        name : str
            The name of the setting to be changed.
            One of

            - ``'sphere_color'``,
            - ``'surface_color'``,
            - ``'mesh_color'``,
            - ``'label_color'``,
            - ``'dot_color'``,
            - ``'cartoon_color'``,
            - ``'ribbon_color'``,
            - ``'transparency'`` (for surfaces) or
            - ``'sphere_transparency'``.

        value : object
            The new value for the given setting name.
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        state : int, optional
            The state to apply the command on.
            By default, the command is applied on all states of this
            *PyMOL* object.
        """
        state = 0 if state is None else state
        self._cmd.set(name, value, self._into_selection(selection), state)

    @validate
    def set_bond(self, name, value, selection1=None, selection2=None, state=None):
        """
        Change per-bond settings for all bonds which exist
        between two atom selections.

        This method is a thin wrapper around the *PyMOL* ``set_bond()``
        command.

        Parameters
        ----------
        name : str
            The name of the setting to be changed.
            One of

            - ``'valence'``,
            - ``'line_width'``,
            - ``'line_color'``,
            - ``'stick_radius'``,
            - ``'stick_color'`` or
            - ``'stick_transparency'``.

        value : object
            The new value for the given setting name.
        selection1, selection2 : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, `selection1` applies to all atoms of this
            *PyMOL* object and `selection2` applies to the same atoms as
            `selection1`.
        state : int, optional
            The state to apply the command on.
            By default, the command is applied on all states of this
            *PyMOL* object.
        """
        state = 0 if state is None else state
        selection2 = selection1 if selection2 is None else selection2
        self._cmd.set_bond(
            name,
            value,
            self._into_selection(selection1),
            self._into_selection(selection2),
            state,
        )

    @validate
    def show(self, representation, selection=None):
        """
        Turn on an atom representation (e.g. sticks, spheres, etc.).

        This method is a thin wrapper around the *PyMOL* ``show()``
        command.

        Parameters
        ----------
        representation : str
            One of

            - ``'lines'``,
            - ``'spheres'``,
            - ``'mesh'``,
            - ``'ribbon'``,
            - ``'cartoon'``,
            - ``'sticks'``,
            - ``'dots'``,
            - ``'surface'``,
            - ``'label'``,
            - ``'extent'``,
            - ``'nonbonded'``,
            - ``'nb_spheres'``,
            - ``'slice'`` or
            - ``'cell'``.

        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        """
        self._cmd.show(representation, self._into_selection(selection))

    @validate
    def show_as(self, representation, selection=None):
        """
        Turn on a representation (e.g. sticks, spheres, etc.) and hide
        all other representations.

        This method is a thin wrapper around the *PyMOL* ``show_as()``
        command.

        Parameters
        ----------
        representation : str
            One of

            - ``'lines'``,
            - ``'spheres'``,
            - ``'mesh'``,
            - ``'ribbon'``,
            - ``'cartoon'``,
            - ``'sticks'``,
            - ``'dots'``,
            - ``'surface'``,
            - ``'label'``,
            - ``'extent'``,
            - ``'nonbonded'``,
            - ``'nb_spheres'``,
            - ``'slice'`` or
            - ``'cell'``.

        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        """
        self._cmd.show_as(representation, self._into_selection(selection))

    @validate
    def smooth(self, selection=None, passes=1, window=5, first=1, last=0, ends=False):
        """
        Perform a moving average over the coordinate states.

        This method is a thin wrapper around the *PyMOL* ``smooth()``
        command.

        Parameters
        ----------
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        passes : int, optional
            The number of smoothing passes.
        window : int, optional
            The size of the moving window.
        first, last : int, optional
            The interval of states to smooth.
        ends : bool, optional
            If set to true, the end states are also smoothed using a
            weighted asymmetric window.
        """
        self._cmd.smooth(
            self._into_selection(selection), passes, window, first, last, int(ends)
        )

    # TODO: def spectrum()

    @validate
    def unset(self, name, selection=None, state=None):
        """
        Clear per-atom settings.

        This method is a thin wrapper around the *PyMOL* ``set()``
        command.

        Parameters
        ----------
        name : str
            The name of the setting to be cleared.
            One of

            - ``'sphere_color'``,
            - ``'surface_color'``,
            - ``'mesh_color'``,
            - ``'label_color'``,
            - ``'dot_color'``,
            - ``'cartoon_color'``,
            - ``'ribbon_color'``,
            - ``'transparency'`` (for surfaces) or
            - ``'sphere_transparency'``.

        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        state : int, optional
            The state to apply the command on.
            By default, the command is applied on all states of this
            *PyMOL* object.
        """
        state = 0 if state is None else state
        self._cmd.unset(name, self._into_selection(selection), state)

    @validate
    def unset_bond(self, name, selection1=None, selection2=None, state=None):
        """
        Clear per-bond settings for all bonds which exist
        between two atom selections.

        This method is a thin wrapper around the *PyMOL* ``unset_bond()``
        command.

        Parameters
        ----------
        name : str
            The name of the setting to be cleared.
            One of

            - ``'valence'``,
            - ``'line_width'``,
            - ``'line_color'``,
            - ``'stick_radius'``,
            - ``'stick_color'`` or
            - ``'stick_transparency'``.

        selection1, selection2 : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, `selection1` applies to all atoms of this
            *PyMOL* object and `selection2` applies to the same atoms as
            `selection1`.
        state : int, optional
            The state to apply the command on.
            By default, the command is applied on all states of this
            *PyMOL* object.
        """
        state = 0 if state is None else state
        selection2 = selection1 if selection2 is None else selection2
        self._cmd.unset_bond(
            name,
            self._into_selection(selection1),
            self._into_selection(selection2),
            state,
        )

    @validate
    def zoom(self, selection=None, buffer=0.0, state=None, complete=False):
        """
        Scale and translate the window and the origin to cover the
        selected atoms.

        This method is a thin wrapper around the *PyMOL* ``zoom()``
        command.

        Parameters
        ----------
        selection : str or int or slice or ndarray, dtype=bool or ndarray, dtype=int, optional
            A *Biotite* compatible atom selection index,
            e.g. a boolean mask, or a *PyMOL* selection expression that
            selects the atoms of this *PyMOL* object to apply the
            command on.
            By default, the command is applied on all atoms of this
            *PyMOL* object.
        buffer : float, optional
            An additional distance to the calculated camera position.
        state : int, optional
            The state to apply the command on.
            By default, the command is applied on all states of this
            *PyMOL* object.
        complete : bool, optional
            If set to true, it is insured that no atoms centers are
            clipped.
        """
        state = 0 if state is None else state
        self._cmd.zoom(
            self._into_selection(selection, True), buffer, state, int(complete)
        )


class NonexistentObjectError(Exception):
    """
    Indicates that a *PyMOL* object with a given name does not exist.
    """

    pass


class ModifiedObjectError(Exception):
    """
    Indicates that a atoms were added to or removed from the *PyMOL*
    object after the corresponding :class:`PyMOLObject` was created.
    """

    pass
