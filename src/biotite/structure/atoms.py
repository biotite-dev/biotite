# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module contains the main types of the ``structure`` subpackage:
:class:`Atom`, :class:`AtomArray` and :class:`AtomArrayStack`. 
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["Atom", "AtomArray", "AtomArrayStack",
           "array", "stack", "repeat", "from_template", "coord"]

import numbers
import abc
import numpy as np
from .bonds import BondList
from ..copyable import Copyable


class _AtomArrayBase(Copyable, metaclass=abc.ABCMeta):
    """
    Private base class for :class:`AtomArray` and
    :class:`AtomArrayStack`.
    It implements functionality for annotation arrays and also
    rudimentarily for coordinates.
    """
    
    def __init__(self, length):
        """
        Create the annotation arrays
        """
        self._annot = {}
        self._array_length = length
        self._coord = None
        self._bonds = None
        self._box = None
        self.add_annotation("chain_id", dtype="U4")
        self.add_annotation("res_id", dtype=int)
        self.add_annotation("ins_code", dtype="U1")
        self.add_annotation("res_name", dtype="U3")
        self.add_annotation("hetero", dtype=bool)
        self.add_annotation("atom_name", dtype="U6")
        self.add_annotation("element", dtype="U2")
        
    def array_length(self):
        """
        Get the length of the atom array.
        
        This value is equivalent to the length of each annotation array.
        For :class:`AtomArray` it is the same as ``len(array)``.
        
        Returns
        -------
        length : int
            Length of the array(s).
        """
        return self._array_length

    @property
    @abc.abstractmethod
    def shape(self):
        """
        Tuple of array dimensions.

        This property contains the current shape of the object.

        Returns
        -------
        shape : tuple of int
            Shape of the object.
        """
        return 
        
    def add_annotation(self, category, dtype):
        """
        Add an annotation category, if not already existing.
        
        Initially the new annotation is filled with the *zero*
        representation of the given type.
        
        Parameters
        ----------
        category : str
            The annotation category to be added.
        dtype : type or str
            A type instance or a valid *NumPy* *dtype* string.
            Defines the type of the annotation
        
        See Also
        --------
        set_annotation
        """
        if category not in self._annot:
            self._annot[str(category)] = np.zeros(self._array_length,
                                                  dtype=dtype)
            
    def del_annotation(self, category):
        """
        Removes an annotation category.
        
        Parameters
        ----------
        category : str
            The annotation category to be removed.
        """
        if category not in self._annot:
            del self._annot[str(category)]
            
    def get_annotation(self, category):
        """
        Return an annotation array.
        
        Parameters
        ----------
        category : str
            The annotation category to be returned.
            
        Returns
        -------
        array : ndarray
            The annotation array.
        """
        if category not in self._annot:
            raise ValueError(
                f"Annotation category '{category}' is not existing"
            )
        return self._annot[category]
    
    def set_annotation(self, category, array):
        """
        Set an annotation array. If the annotation category does not
        exist yet, the category is created.
        
        Parameters
        ----------
        category : str
            The annotation category to be set.
        array : ndarray or None
            The new value of the annotation category. The size of the
            array must be the same as the array length.
        """
        if len(array) != self._array_length:
            raise IndexError(
                f"Expected array length {self._array_length}, "
                f"but got {len(array)}"
            )
        self._annot[category] = np.asarray(array)
        
    def get_annotation_categories(self):
        """
        Return a list containing all annotation array categories.
            
        Returns
        -------
        categories : list
            The list containing the names of each annotation array.
        """
        return list(self._annot.keys())
            
    def _subarray(self, index):
        # Index is one dimensional (boolean mask, index array)
        new_coord = self._coord[..., index, :]
        new_length = new_coord.shape[-2]
        if isinstance(self, AtomArray):
            new_object = AtomArray(new_length)
        elif isinstance(self, AtomArrayStack):
            new_depth = new_coord.shape[-3]
            new_object = AtomArrayStack(new_depth, new_length)
        new_object._coord = new_coord
        if self._bonds is not None:
            new_object._bonds = self._bonds[index]
        if self._box is not None:
            new_object._box = self._box
        for annotation in self._annot:
            new_object._annot[annotation] = (self._annot[annotation]
                                             .__getitem__(index))
        return new_object
        
    def _set_element(self, index, atom):
        try:
            if isinstance(index, (numbers.Integral, np.ndarray)):
                for name in self._annot:
                    self._annot[name][index] = atom._annot[name]
                self._coord[..., index, :] = atom.coord
            else:
                raise TypeError(
                    f"Index must be integer, not '{type(index).__name__}'"
                )
        except KeyError:
            raise KeyError("The annotations of the 'Atom' are incompatible")
        
    def _del_element(self, index):
        if isinstance(index, numbers.Integral):
            for name in self._annot:
                self._annot[name] = np.delete(self._annot[name], index, axis=0)
            self._coord = np.delete(self._coord, index, axis=-2)
            self._array_length = self._coord.shape[-2]
            if self._bonds is not None:
                mask = np.ones(self._bonds.get_atom_count(), dtype=bool)
                mask[index] = False
                self._bonds = self._bonds[mask]
        else:
            raise TypeError(
                    f"Index must be integer, not '{type(index).__name__}'"
                )
    
    def equal_annotations(self, item):
        """
        Check, if this object shares equal annotation arrays with the
        given :class:`AtomArray` or :class:`AtomArrayStack`.
        
        Parameters
        ----------
        item : AtomArray or AtomArrayStack
            The object to compare the annotation arrays with.
        
        Returns
        -------
        equality : bool
            True, if the annotation arrays are equal.
        """
        if not isinstance(item, _AtomArrayBase):
            return False
        if not self.equal_annotation_categories(item):
            return False
        for name in self._annot:
            if not np.array_equal(self._annot[name], item._annot[name]):
                return False
        return True
    
    def equal_annotation_categories(self, item):
        """
        Check, if this object shares equal annotation array categories
        with the given :class:`AtomArray` or :class:`AtomArrayStack`.
        
        Parameters
        ----------
        item : AtomArray or AtomArrayStack
            The object to compare the annotation arrays with.
        
        Returns
        -------
        equality : bool
            True, if the annotation array names are equal.
        """
        return sorted(self._annot.keys()) == sorted(item._annot.keys())
    
    def __getattr__(self, attr):
        """
        If the attribute is an annotation, the annotation is returned
        from the dictionary.
        Exposes coordinates.
        """
        if attr == "coord":
            return self._coord
        if attr == "bonds":
            return self._bonds
        if attr == "box":
            return self._box
        elif attr in self._annot:
            return self._annot[attr]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )
        
    def __setattr__(self, attr, value):
        """
        If the attribute is an annotation, the :attr:`value` is saved
        to the annotation in the dictionary.
        Exposes coordinates.
        :attr:`value` must have same length as :func:`array_length()`.
        """
        if attr == "coord":
            if not isinstance(value, np.ndarray):
                raise TypeError("Value must be ndarray of floats")
            if isinstance(self, AtomArray):
                if value.ndim != 2:
                    raise ValueError(
                        "A 2-dimensional ndarray is expected "
                        "for an AtomArray"
                )
            elif isinstance(self, AtomArrayStack):
                if value.ndim != 3:
                    raise ValueError(
                        "A 3-dimensional ndarray is expected "
                        "for an AtomArrayStack"
                )
            if value.shape[-2] != self._array_length:
                raise ValueError(
                    f"Expected array length {self._array_length}, "
                    f"but got {len(value)}"
                )
            if value.shape[-1] != 3:
                raise TypeError("Expected 3 coordinates for each atom")
            self._coord = value.astype(np.float32, copy=False)
        
        elif attr == "bonds":
            if isinstance(value, BondList):
                if value.get_atom_count() != self._array_length:
                    raise ValueError(
                        f"Array length is {self._array_length}, "
                        f"but bond list has {value.get_atom_count()} atoms"
                    )
                self._bonds = value
            elif value is None:
                # Remove bond list
                self._bonds = None
            else:
                raise TypeError("Value must be 'BondList'")
        
        elif attr == "box":
            if value is None:
                self._box = None
            elif isinstance(self, AtomArray):
                if value.ndim != 2:
                    raise ValueError(
                        "A 2-dimensional ndarray is expected "
                        "for an AtomArray"
                )
            elif isinstance(self, AtomArrayStack):
                if value.ndim != 3:
                    raise ValueError(
                        "A 3-dimensional ndarray is expected "
                        "for an AtomArrayStack"
                )
            if isinstance(value, np.ndarray):
                if value.shape[-2:] != (3,3):
                    raise TypeError("Box must be a 3x3 matrix (three vectors)")
                self._box = value.astype(np.float32, copy=False)
            elif value is None:
                # Remove box
                self._box = None
            else:
                raise TypeError("Box must be ndarray of floats or None")
        
        # This condition is required, since otherwise 
        # call of the next one would result
        # in indefinite calls of __setattr__
        elif attr == "_annot":
            super().__setattr__(attr, value)
        elif attr in self._annot:
            self.set_annotation(attr, value)
        else:
            super().__setattr__(attr, value)
            
    def __dir__(self):
        attr = super().__dir__()
        attr.append("coord")
        attr.append("bonds")
        attr.append("box")
        for name in self._annot.keys():
            attr.append(name)
        return attr
    
    def __eq__(self, item):
        """
        See Also
        --------
        equal_annotations
        """
        if not self.equal_annotations(item):
            return False
        if self._bonds != item._bonds:
            return False
        if self._box is None:
            if item._box is not None:
                return False
        else:
            if not np.array_equal(self._box, item._box):
                return False
        return np.array_equal(self._coord, item._coord)
    
    def __len__(self):
        """
        The length of the annotation arrays.
        
        Returns
        -------
        length : int
            Length of the annotation arrays.
        """
        return self._array_length
    
    def __add__(self, array):
        if type(self) != type(array):
            raise TypeError("Can only concatenate two arrays or two stacks")
        # Create either new array or stack, depending of the own type
        if isinstance(self, AtomArray):
            concat = AtomArray(length = self._array_length+array._array_length)
        if isinstance(self, AtomArrayStack):
            concat = AtomArrayStack(self.stack_depth(),
                                    self._array_length + array._array_length)
        
        concat._coord = np.concatenate((self._coord, array.coord), axis=-2)
        
        # Transfer only annotations,
        # which are existent in both operands
        arr_categories = list(array._annot.keys())
        for category in self._annot.keys():
            if category in arr_categories:
                annot = self._annot[category]
                arr_annot = array._annot[category]
                concat._annot[category] = np.concatenate((annot,arr_annot))
        
        # Concatenate bonds lists,
        # if at least one of them contains bond information
        if self._bonds is not None or array._bonds is not None:
            bonds1 = self._bonds
            bonds2 = array._bonds
            if bonds1 is None:
               bonds1 = BondList(self._array_length)
            if bonds2 is None:
                bonds2 = BondList(array._array_length)
            concat._bonds = bonds1 + bonds2
        
        # Copy box
        if self._box is not None:
            concat._box = np.copy(self._box)
        return concat
    
    def __copy_fill__(self, clone):
        super().__copy_fill__(clone)
        self._copy_annotations(clone)
        clone._coord = np.copy(self._coord)
    
    def _copy_annotations(self, clone):
        for name in self._annot:
            clone._annot[name] = np.copy(self._annot[name])
        if self._box is not None:
            clone._box = np.copy(self._box)
        if self._bonds is not None:
            clone._bonds = self._bonds.copy()
    

class Atom(Copyable):
    """
    A representation of a single atom.
    
    The coordinates an annotations can be accessed directly.
    A detailed description of each annotation category can be viewed
    :doc:`here </apidoc/biotite.structure>`.
    
    Parameters
    ----------
    coord: list or ndarray
        The x, y and z coordinates.
    kwargs
        Atom annotations as key value pair.
    
    Attributes
    ----------
    {annot} : scalar
        Annotations for this atom.
    coord : ndarray, dtype=float
        ndarray containing the x, y and z coordinate of the atom.
    shape : tuple of int
        Shape of the object.
        In case of an :class:`Atom`, the tuple is empty.
    
    Examples
    --------
    
    >>> atom = Atom([1,2,3], chain_id="A")
    >>> atom.atom_name = "CA"
    >>> print(atom.atom_name)
    CA
    >>> print(atom.coord)
    [1. 2. 3.]
        
    """
    
    def __init__(self, coord, **kwargs):
        self._annot = {}
        self._annot["chain_id"] = ""
        self._annot["res_id"] = 0
        self._annot["ins_code"] = ""
        self._annot["res_name"] = ""
        self._annot["hetero"] = False
        self._annot["atom_name"] = ""
        self._annot["element"] = ""
        if "kwargs" in kwargs:
            # kwargs are given directly as dictionary
            kwargs = kwargs["kwargs"]
        for name, annotation in kwargs.items():
            self._annot[name] = annotation
        coord = np.array(coord, dtype=np.float32)
        # Check if coord contains x,y and z coordinates
        if coord.shape != (3,):
            raise ValueError("Position must be ndarray with shape (3,)")
        self.coord = coord
    
    @property
    def shape(self):
        return ()
        
    def __getattr__(self, attr):
        if attr in self._annot:
            return self._annot[attr]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )
        
    def __setattr__(self, attr, value):
        # First condition is required, since call of the second would
        # result in indefinite calls of __getattr__
        if attr == "_annot":
            super().__setattr__(attr, value)
        elif attr in self._annot:
            self._annot[attr] = value
        else:
            super().__setattr__(attr, value)
    
    def __str__(self):
        hetero = "HET" if self.hetero else ""
        return f"{hetero:3} {self.chain_id:3} " \
               f"{self.res_id:5d}{self.ins_code:1} {self.res_name:3} " \
               f"{self.atom_name:6} {self.element:2}     " \
               f"{self.coord[0]:8.3f} " \
               f"{self.coord[1]:8.3f} " \
               f"{self.coord[2]:8.3f}"
    
    def __eq__(self, item):
        if not isinstance(item, Atom):
            return False
        if not np.array_equal(self.coord, item.coord):
            return False
        if self._annot.keys() != item._annot.keys():
            return False
        for name in self._annot:
            if self._annot[name] != item._annot[name]:
                return False
        return True
    
    def __ne__(self, item):
        return not self == item
    
    def __copy_create__(self):
        return Atom(self.coord, **self._annot)

    
class AtomArray(_AtomArrayBase):
    """
    An array representation of a model consisting of multiple atoms.
    
    An :class:`AtomArray` can be seen as a list of :class:`Atom`
    instances.
    Instead of using directly a list, this class uses an *NumPy*
    :class:`ndarray` for each annotation category and the coordinates.
    These
    coordinates can be accessed directly via the :attr:`coord`
    attribute.
    The annotations are accessed either via the category as attribute
    name or the :func:`get_annotation()`, :func:`set_annotation()`
    method.
    Usage of custom annotations is achieved via :func:`add_annotation()`
    or :func:`set_annotation()`.
    A detailed description of each annotation category can be viewed
    :doc:`here </apidoc/biotite.structure>`.
    
    In order to get an an subarray of an :class:`AtomArray`,
    *NumPy* style indexing is used.
    This includes slices, boolean arrays, index arrays and even
    *Ellipsis* notation.
    Using a single integer as index returns a single :class:`Atom`
    instance.
    
    Inserting or appending an :class:`AtomArray` to another
    :class:`AtomArray` is done with the '+' operator.
    Only the annotation categories, which are existing in both arrays,
    are transferred to the new array.

    Optionally, an :class:`AtomArray` can store chemical bond
    information via a :class:`BondList` object.
    It can be accessed using the :attr:`bonds` attribute.
    If no bond information is available, :attr:`bonds` is ``None``.
    Consequently the bond information can be removed from the
    :class:`AtomArray`, by setting :attr:`bonds` to ``None``.
    When indexing the :class:`AtomArray` the atom indices in the
    associated :class:`BondList` are updated as well, hence the indices
    in the :class:`BondList` will always point to the same atoms.
    If two :class:`AtomArray` instances are concatenated, the resulting
    :class:`AtomArray` will contain the merged :class:`BondList` if at
    least one of the operands contains bond information.

    The :attr:`box` attribute contains the box vectors of the unit cell
    or the MD simulation box, respectively.
    Hence, it is a *3 x 3* *ndarray* with the vectors in the last
    dimension.
    If no box is provided, the attribute is ``None``.
    Setting the :attr:`box` attribute to ``None`` means removing the
    box from the atom array.

    Parameters
    ----------
    length : int
        The fixed amount of atoms in the array.
    
    Attributes
    ----------
    {annot} : ndarray
        Multiple n-length annotation arrays.
    coord : ndarray, dtype=float, shape=(n,3)
        ndarray containing the x, y and z coordinate of the
        atoms.
    bonds: BondList or None
        A :class:`BondList`, specifying the indices of atoms
        that form a chemical bond.
    box: ndarray, dtype=float, shape=(3,3) or None
        The surrounding box. May represent a MD simulation box
        or a crystallographic unit cell.
    shape : tuple of int
        Shape of the atom array.
        The single value in the tuple is
        the length of the atom array.
    
    Examples
    --------
    Creating an atom array from atoms:
    
    >>> atom1 = Atom([1,2,3], chain_id="A")
    >>> atom2 = Atom([2,3,4], chain_id="A")
    >>> atom3 = Atom([3,4,5], chain_id="B")
    >>> atom_array = array([atom1, atom2, atom3])
    >>> print(atom_array.array_length())
    3
    
    Accessing an annotation array:
    
    >>> print(atom_array.chain_id)
    ['A' 'A' 'B']
        
    Accessing the coordinates:
    
    >>> print(atom_array.coord)
    [[1. 2. 3.]
     [2. 3. 4.]
     [3. 4. 5.]]
    
    *NumPy* style filtering:
    
    >>> atom_array = atom_array[atom_array.chain_id == "A"]
    >>> print(atom_array.array_length())
    2
        
    Inserting an atom array:
        
    >>> insert = array([Atom([7,8,9], chain_id="C")])
    >>> atom_array = atom_array[0:1] + insert + atom_array[1:2]
    >>> print(atom_array.chain_id)
    ['A' 'C' 'A']
    """
    
    def __init__(self, length):
        super().__init__(length)
        if length is None:
            self._coord = None
        else:
            self._coord = np.full((length, 3), np.nan, dtype=np.float32)
    
    @property
    def shape(self):
        """
        Tuple of array dimensions.

        This property contains the current shape of the
        :class:`AtomArray`.

        Returns
        -------
        shape : tuple of int
            Shape of the array.
            The single value in the tuple is
            the :func:`array_length()`.

        See Also
        --------
        array_length
        """
        return self.array_length(),

    def get_atom(self, index):
        """
        Obtain the atom instance of the array at the specified index.
        
        The same as ``array[index]``, if `index` is an integer.
        
        Parameters
        ----------
        index : int
            Index of the atom.
        
        Returns
        -------
        atom : Atom
            Atom at position `index`. 
        """
        kwargs = {}
        for name, annotation in self._annot.items():
            kwargs[name] = annotation[index]
        return Atom(coord = self._coord[index], kwargs=kwargs)
    
    def __iter__(self):
        """
        Iterate through the array.
        
        Yields
        ------
        atom : Atom
        """
        i = 0
        while i < len(self):
            yield self.get_atom(i)
            i += 1
    
    def __getitem__(self, index):
        """
        Obtain a subarray or the atom instance at the specified index.
        
        Parameters
        ----------
        index : object
            All index types *NumPy* accepts, are valid.
        
        Returns
        -------
        sub_array : Atom or AtomArray
            If `index` is an integer an :class:`Atom` instance is
            returned.
            Otherwise an :class:`AtomArray` with reduced length is
            returned.
        """
        if isinstance(index, numbers.Integral):
            return self.get_atom(index)
        elif isinstance(index, tuple):
            if len(index) == 2 and index[0] is Ellipsis:
                # If first index is "...", just ignore the first index
                return self.__getitem__(index[1])
            else:
                raise IndexError(
                    "'AtomArray' does not accept multidimensional indices"
                )
        else:
            return self._subarray(index)
        
    def __setitem__(self, index, atom):
        """
        Set the atom at the specified array position.
        
        Parameters
        ----------
        index : int
            The position, where the atom is set.
        atom : Atom
            The atom to be set.
        """
        self._set_element(index, atom)
        
    def __delitem__(self, index):
        """
        Deletes the atom at the specified array position.
        
        Parameters
        ----------
        index : int
            The position where the atom should be deleted.
        """
        self._del_element(index)
        
    def __len__(self):
        """
        The length of the array.
        
        Returns
        -------
        length : int
            Length of the array.
        """
        return self.array_length()
    
    def __eq__(self, item):
        """
        Check if the array equals another :class:`AtomArray`.
        
        Parameters
        ----------
        item : object
            Object to campare the array with.
        
        Returns
        -------
        equal : bool
            True, if `item` is an :class:`AtomArray`
            and all its attribute arrays equals the ones of this object.
        """
        if not super().__eq__(item):
            return False
        if not isinstance(item, AtomArray):
            return False
        return True
    
    def __str__(self):
        """
        Get a string representation of the array.
        
        Each line contains the attributes of one atom.
        """
        return "\n".join([str(atom) for atom in self])
    
    def __copy_create__(self):
        return AtomArray(self.array_length())


class AtomArrayStack(_AtomArrayBase):
    """
    A collection of multiple :class:`AtomArray` instances, where each
    atom array has equal annotation arrays.
    
    Effectively, this means that each atom is occuring in every array in
    the stack at differing coordinates. This situation arises e.g. in
    NMR-elucidated or simulated structures. Since the annotations are
    equal for each array, the annotation arrays are 1-D, while the
    coordinate array is 3-D (m x n x 3).
    A detailed description of each annotation category can be viewed
    :doc:`here </apidoc/biotite.structure>`.
    
    Indexing works similar to :class:`AtomArray`, with the difference,
    that two index dimensions are possible:
    The first index dimension specifies the array(s), the second index
    dimension specifies the atoms in each array (same as the index
    in :class:`AtomArray`).
    Using a single integer as first dimension index returns a single
    :class:`AtomArray` instance.
    
    Concatenation of atoms for each array in the stack is done using the
    '+' operator. For addition of atom arrays onto the stack use the
    :func:`stack()` method.

    The :attr:`box` attribute has the shape *m x 3 x 3*, as the cell
    might be different for each frame in the atom array stack.
    
    Parameters
    ----------
    depth : int
        The fixed amount of arrays in the stack. When indexing, this is
        the length of the first dimension.
        
    length : int
        The fixed amount of atoms in each array in the stack. When
        indexing, this is the length of the second dimension.
    
    Attributes
    ----------
    {annot} : ndarray, shape=(n,)
        Mutliple n-length annotation arrays.
    coord : ndarray, dtype=float, shape=(m,n,3)
        ndarray containing the x, y and z coordinate of the
        atoms.
    bonds: BondList or None
        A :class:`BondList`, specifying the indices of atoms
        that form a chemical bond.
    box: ndarray, dtype=float, shape=(m,3,3) or None
        The surrounding box. May represent a MD simulation box
        or a crystallographic unit cell.
    shape : tuple of int
        Shape of the stack.
        The numbers correspond to the stack depth
        and array length, respectively.
    
    See also
    --------
    AtomArray
    
    Examples
    --------
    Creating an atom array stack from two arrays:
    
    >>> atom1 = Atom([1,2,3], chain_id="A")
    >>> atom2 = Atom([2,3,4], chain_id="A")
    >>> atom3 = Atom([3,4,5], chain_id="B")
    >>> atom_array1 = array([atom1, atom2, atom3])
    >>> print(atom_array1.coord)
    [[1. 2. 3.]
     [2. 3. 4.]
     [3. 4. 5.]]
    >>> atom_array2 = atom_array1.copy()
    >>> atom_array2.coord += 3
    >>> print(atom_array2.coord)
    [[4. 5. 6.]
     [5. 6. 7.]
     [6. 7. 8.]]
    >>> array_stack = stack([atom_array1, atom_array2])
    >>> print(array_stack.coord)
    [[[1. 2. 3.]
      [2. 3. 4.]
      [3. 4. 5.]]
    <BLANKLINE>
     [[4. 5. 6.]
      [5. 6. 7.]
      [6. 7. 8.]]]
    """
    
    def __init__(self, depth, length):
        super().__init__(length)
        if depth == None or length == None:
            self._coord = None
        else:
            self._coord = np.full((depth, length, 3), np.nan, dtype=np.float32)
    
    def get_array(self, index):
        """
        Obtain the atom array instance of the stack at the specified
        index.
        
        The same as ``stack[index]``, if `index` is an integer.
        
        Parameters
        ----------
        index : int
            Index of the atom array.
        
        Returns
        -------
        array : AtomArray
            AtomArray at position `index`. 
        """
        array = AtomArray(self.array_length())
        for name in self._annot:
            array._annot[name] = self._annot[name]
        array._coord = self._coord[index]
        if self._bonds is not None:
            array._bonds = self._bonds.copy()
        if self._box is not None:
            array._box = self._box[index]

        return array
    
    def stack_depth(self):
        """
        Get the depth of the stack.
        
        This value represents the amount of atom arrays in the stack.
        It is the same as ``len(array)``.
        
        Returns
        -------
        length : int
            Length of the array(s).
        """
        return len(self)

    @property
    def shape(self):
        """
        Tuple of array dimensions.

        This property contains the current shape of the
        :class:`AtomArrayStack`.

        Returns
        -------
        shape : tuple of int
            Shape of the stack.
            The numbers correspond to the :func:`stack_depth()`
            and :func:`array_length()`, respectively.
        """
        return self.stack_depth(), self.array_length()

    def __iter__(self):
        """
        Iterate through the array.
        
        Yields
        ------
        array : AtomArray
        """
        i = 0
        while i < len(self):
            yield self.get_array(i)
            i += 1
            
    def __getitem__(self, index):
        """
        Obtain the atom array instance or an substack at the specified
        index.
        
        Parameters
        ----------
        index : object
            All index types *NumPy* accepts are valid.
        
        Returns
        -------
        sub_array : AtomArray or AtomArrayStack
            If `index` is an integer an :class:`AtomArray` instance is
            returned.
            Otherwise an :class:`AtomArrayStack` with reduced depth and
            length is returned.
            In case the index is a tuple(int, int) an :class:`Atom`
            instance is returned.  
        """
        if isinstance(index, numbers.Integral):
            return self.get_array(index)
        elif isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError(
                    "'AtomArrayStack' does not accept an index "
                    "with more than two dimensions"
                )
            if isinstance(index[0], numbers.Integral):
                array = self.get_array(index[0])
                return array.__getitem__(index[1])
            else:
                if isinstance(index[1], numbers.Integral):
                    # Prevent reduction in dimensionality
                    # in second dimension
                    new_stack = self._subarray(slice(index[1], index[1]+1))
                else:
                    new_stack = self._subarray(index[1])
                if index[0] is not Ellipsis:
                    new_stack._coord = new_stack._coord[index[0]]
                    if new_stack._box is not None:
                        new_stack._box = new_stack._box[index[0]]
                return new_stack
        else:
            new_stack = AtomArrayStack(depth=0, length=self.array_length())
            self._copy_annotations(new_stack)
            new_stack._coord = self._coord[index]
            if self._box is not None:
                new_stack._box = self._box[index]
            return new_stack
            
    
    def __setitem__(self, index, array):
        """
        Set the atom array at the specified stack position.
        
        The array and the stack must have equal annotation arrays.
        
        Parameters
        ----------
        index : int
            The position, where the array atom is set.
        array : AtomArray
            The atom array to be set.
        """
        if not self.equal_annotations(array):
            raise ValueError(
                "The stack and the array have unequal annotations"
            )
        if self.bonds != array.bonds:
            raise ValueError(
                "The stack and the array have unequal bonds"
            )
        if isinstance(index, numbers.Integral):
            self.coord[index] = array.coord
            if self.box is not None:
                self.box[index] = array.box
        else:
            raise TypeError(
                f"Index must be integer, not '{type(index).__name__}'"
            )
        
    def __delitem__(self, index):
        """
        Deletes the atom array at the specified stack position.
        
        Parameters
        ----------
        index : int
            The position where the atom array should be deleted.
        """
        if isinstance(index, numbers.Integral):
            self._coord = np.delete(self._coord, index, axis=0)
        else:
            raise TypeError(
                f"Index must be integer, not '{type(index).__name__}'"
            )
    
    def __len__(self):
        """
        The depth of the stack, i.e. the amount of models.
        
        Returns
        -------
        depth : int
            depth of the array.
        """
        # length is determined by length of coord attribute
        return self._coord.shape[0]
    
    def __eq__(self, item):
        """
        Check if the array equals another :class:`AtomArray`
        
        Parameters
        ----------
        item : object
            Object to campare the array with.
        
        Returns
        -------
        equal : bool
            True, if `item` is an :class:`AtomArray`
            and all its attribute arrays equals the ones of this object.
        """
        if not super().__eq__(item):
            return False
        if not isinstance(item, AtomArrayStack):
            return False
        return True
    
    def __str__(self):
        """
        Get a string representation of the stack.
        
        :class:`AtomArray` strings eparated by blank lines
        and a line indicating the index.
        """
        string = ""
        for i, array in enumerate(self):
            string += "Model " + str(i+1) + "\n"
            string += str(array) + "\n" + "\n"
        return string
    
    def __copy_create__(self):
        return AtomArrayStack(self.stack_depth(), self.array_length())


def array(atoms):
    """
    Create an :class:`AtomArray` from a list of :class:`Atom`.
    
    Parameters
    ----------
    atoms : iterable object of Atom
        The atoms to be combined in an array.
        All atoms must share the same annotation categories.
    
    Returns
    -------
    array : AtomArray
        The listed atoms as array.
    
    Examples
    --------
    
    Creating an atom array from atoms:
    
    >>> atom1 = Atom([1,2,3], chain_id="A")
    >>> atom2 = Atom([2,3,4], chain_id="A")
    >>> atom3 = Atom([3,4,5], chain_id="B")
    >>> atom_array = array([atom1, atom2, atom3])
    >>> print(atom_array)
        A       0                       1.000    2.000    3.000
        A       0                       2.000    3.000    4.000
        B       0                       3.000    4.000    5.000
    """
    # Check if all atoms have the same annotation names
    # Equality check requires sorting
    names = sorted(atoms[0]._annot.keys())
    for i, atom in enumerate(atoms):
        if sorted(atom._annot.keys()) != names:
            raise ValueError(
                f"The atom at index {i} does not share the same "
                f"annotation categories as the atom at index 0"
            )
    # Add all atoms to AtomArray
    array = AtomArray(len(atoms))
    for i in range(len(atoms)):
        for name in names:
            array._annot[name][i] = atoms[i]._annot[name]
        array._coord[i] = atoms[i].coord
    return array


def stack(arrays):
    """
    Create an :class:`AtomArrayStack` from a list of :class:`AtomArray`.
    
    Parameters
    ----------
    arrays : iterable object of AtomArray
        The atom arrays to be combined in a stack.
        All atom arrays must have an equal number of atoms and equal
        annotation arrays.
    
    Returns
    -------
    stack : AtomArrayStack
        The stacked atom arrays.
    
    Examples
    --------
    Creating an atom array stack from two arrays:
    
    >>> atom1 = Atom([1,2,3], chain_id="A")
    >>> atom2 = Atom([2,3,4], chain_id="A")
    >>> atom3 = Atom([3,4,5], chain_id="B")
    >>> atom_array1 = array([atom1, atom2, atom3])
    >>> print(atom_array1.coord)
    [[1. 2. 3.]
     [2. 3. 4.]
     [3. 4. 5.]]
    >>> atom_array2 = atom_array1.copy()
    >>> atom_array2.coord += 3
    >>> print(atom_array2.coord)
    [[4. 5. 6.]
     [5. 6. 7.]
     [6. 7. 8.]]
    >>> array_stack = stack([atom_array1, atom_array2])
    >>> print(array_stack.coord)
    [[[1. 2. 3.]
      [2. 3. 4.]
      [3. 4. 5.]]
    <BLANKLINE>
     [[4. 5. 6.]
      [5. 6. 7.]
      [6. 7. 8.]]]
    """
    array_count = 0
    ref_array = None
    for i, array in enumerate(arrays):
        if ref_array is None:
            ref_array = array
        array_count += 1
        # Check if all arrays share equal annotations
        if not array.equal_annotations(ref_array):
            raise ValueError(
                f"The annotations of the atom array at index {i} are not "
                f"equal to the annotations of the atom array at index 0"
            )
    array_stack = AtomArrayStack(array_count, ref_array.array_length())
    for name, annotation in ref_array._annot.items():
        array_stack._annot[name] = annotation
    coord_list = [array._coord for array in arrays] 
    array_stack._coord = np.stack(coord_list, axis=0)
    # Take bond list from first array
    array_stack._bonds = ref_array._bonds
    # When all atom arrays provide a box, copy the boxes
    if all([array.box is not None for array in arrays]):
        array_stack.box = np.array([array.box for array in arrays])
    return array_stack


def repeat(atoms, coord):
    """
    Repeat atoms (:class:`AtomArray` or :class:`AtomArrayStack`)
    multiple times in the same model with different coordinates.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or AtomArrayStack, shape=(m,n)
        The atoms to be repeated.
    coord : ndarray, dtype=float, shape=(k,n,3) or shape=(k,m,n,3)
        The coordinates to be used fr the repeated atoms.
        The length of first dimension determines the number of repeats.
        If `atoms` is an :class:`AtomArray` 3 dimensions, otherwise
        4 dimensions are required.
    
    Returns
    -------
    repeated: AtomArray, shape=(n*k,) or AtomArrayStack, shape=(m,n*k)
        The repeated atoms.
        Whether an :class:`AtomArray` or an :class:`AtomArrayStack` is
        returned depends on the input `atoms`.
    
    Examples
    --------

    >>> atoms = array([
    ...     Atom([1,2,3], res_id=1, atom_name="N"),
    ...     Atom([4,5,6], res_id=1, atom_name="CA"),
    ...     Atom([7,8,9], res_id=1, atom_name="C")
    ... ])
    >>> print(atoms)
                1      N                1.000    2.000    3.000
                1      CA               4.000    5.000    6.000
                1      C                7.000    8.000    9.000
    >>> repeat_coord = np.array([
    ...     [[0,0,0], [1,1,1], [2,2,2]],
    ...     [[3,3,3], [4,4,4], [5,5,5]]
    ... ])
    >>> print(repeat(atoms, repeat_coord))
                1      N                0.000    0.000    0.000
                1      CA               1.000    1.000    1.000
                1      C                2.000    2.000    2.000
                1      N                3.000    3.000    3.000
                1      CA               4.000    4.000    4.000
                1      C                5.000    5.000    5.000
    """
    if isinstance(atoms, AtomArray) and coord.ndim != 3:
        raise ValueError(
            f"Expected 3 dimensions for the coordinate array, got {coord.ndim}"
        )
    elif isinstance(atoms, AtomArrayStack) and coord.ndim != 4:
        raise ValueError(
            f"Expected 4 dimensions for the coordinate array, got {coord.ndim}"
        )
    
    repetitions = len(coord)
    orig_length = atoms.array_length()
    new_length = orig_length * repetitions

    if isinstance(atoms, AtomArray):
        if coord.ndim != 3:
            raise ValueError(
                f"Expected 3 dimensions for the coordinate array, "
                f"but got {coord.ndim}"
            )
        repeated = AtomArray(new_length)
        repeated.coord = coord.reshape((new_length, 3))

    elif isinstance(atoms, AtomArrayStack):
        if coord.ndim != 4:
            raise ValueError(
                f"Expected 4 dimensions for the coordinate array, "
                f"but got {coord.ndim}"
            )
        repeated = AtomArrayStack(atoms.stack_depth(), new_length)
        repeated.coord = coord.reshape((atoms.stack_depth(), new_length, 3))
    
    else:
        raise TypeError(
            f"Expected 'AtomArray' or 'AtomArrayStack', "
            f"but got {type(atoms).__name__}"
        )
    
    for category in atoms.get_annotation_categories():
        annot = np.tile(atoms.get_annotation(category), repetitions)
        repeated.set_annotation(category, annot)
    if atoms.bonds is not None:
        bonds = atoms.bonds
        for _ in range(repetitions-1):
            bonds += atoms.bonds
        repeated.bonds = bonds
    if atoms.box is not None:
        repeated.box = atoms.box.copy()
    
    return repeated


def from_template(template, coord, box=None):
    """
    Create an :class:`AtomArrayStack` using template atoms and given
    coordinates.
    
    Parameters
    ----------
    template : AtomArray, shape=(n,) or AtomArrayStack, shape=(m,n)
        The annotation arrays and bonds of the returned stack are taken
        from this template.
    coord : ndarray, dtype=float, shape=(l,n,3)
        The coordinates for each model of the returned stack.
    box : ndarray, optional, dtype=float, shape=(l,3,3)
        The box for each model of the returned stack.
    
    Returns
    -------
    array_stack : AtomArrayStack
        A stack containing the annotation arrays and bonds from
        `template` but the coordinates from `coord` and the boxes from
        `boxes`.
    """
    if template.array_length() != coord.shape[-2]:
        raise ValueError(
            f"Template has {template.array_length()} atoms, but "
            f"{coord.shape[-2]} coordinates are given"
        )

    # Create empty stack with no models
    new_stack = AtomArrayStack(0, template.array_length())
    
    for category in template.get_annotation_categories():
        annot = template.get_annotation(category)
        new_stack.set_annotation(category, annot)
    if template.bonds is not None:
        new_stack.bonds = template.bonds.copy()
    if box is not None:
        new_stack.box = box.copy()
    
    # After setting the coordinates the number of models is the number
    # of models in the new coordinates
    new_stack.coord = coord
    
    return new_stack


def coord(item):
    """
    Get the atom coordinates of the given array.
    
    This may be directly and :class:`Atom`, :class:`AtomArray` or
    :class:`AtomArrayStack` or
    alternatively an (n x 3) or (m x n x 3)  :class:`ndarray`
    containing the coordinates.
    
    Parameters
    ----------
    item : Atom or AtomArray or AtomArrayStack or ndarray
        Returns the :attr:`coord` attribute, if `item` is an
        :class:`Atom`, :class:`AtomArray` or :class:`AtomArrayStack`.
        Directly returns the input, if `item` is an :class:`ndarray`.
    
    Returns
    -------
    coord : ndarray
        Atom coordinates.
    """

    if type(item) in (Atom, AtomArray, AtomArrayStack):
        return item.coord
    elif isinstance(item, np.ndarray):
        return item.astype(np.float32, copy=False)
    else:
        return np.array(item, dtype=np.float32)
