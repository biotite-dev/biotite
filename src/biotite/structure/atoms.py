# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module contains the main types of the `Structure` subpackage:
`Atom`, `AtomArray` and `AtomArrayStack`. 
"""

__author__ = "Patrick Kunzmann"
__all__ = ["Atom", "AtomArray", "AtomArrayStack", "array", "stack", "coord"]

import numbers
import abc
import numpy as np
from .bonds import BondList
from ..copyable import Copyable


class _AtomArrayBase(Copyable, metaclass=abc.ABCMeta):
    """
    Private base class for `AtomArray` and `AtomArrayStack`. It
    implements functionality for annotation arrays and also
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
        self.add_annotation("chain_id", dtype="U3")
        self.add_annotation("res_id", dtype=int)
        self.add_annotation("res_name", dtype="U3")
        self.add_annotation("hetero", dtype=bool)
        self.add_annotation("atom_name", dtype="U6")
        self.add_annotation("element", dtype="U2")
        
    def array_length(self):
        """
        Get the length of the atom array.
        
        This value is equivalent to the length of each annotation array.
        For `AtomArray` it is the same as ``len(array)``.
        
        Returns
        -------
        length : int
            Length of the array(s).
        """
        return self._array_length
        
    def add_annotation(self, category, dtype):
        """
        Add an annotation category, if not already existing.
        
        Initially the new annotation is filled with the `zero`
        representation of the given type.
        
        Parameters
        ----------
        category : str
            The annotation category to be added.
        dtype : type or str
            A type instance or a valid `NumPy` `dtype` string.
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
        if not isinstance(array, np.ndarray):
            raise TypeError("Annotation must be an 'ndarray'")
        if len(array) != self._array_length:
            raise IndexError(
                f"Expected array length {self._array_length}, "
                f"but got {len(array)}"
            )
        self._annot[category] = array
        
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
        for annotation in self._annot:
            new_object._annot[annotation] = (self._annot[annotation]
                                             .__getitem__(index))
        return new_object
        
    def _set_element(self, index, atom):
        try:
            if isinstance(index, numbers.Integral):
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
        given `AtomArray` or `AtomArrayStack`.
        
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
        Check, if this object shares equal annotation array catgeories
        with the given `AtomArray` or `AtomArrayStack`.
        
        Parameters
        ----------
        item : AtomArray or AtomArrayStack
            The object to compare the annotation arrays with.
        
        Returns
        -------
        equality : bool
            True, if the annotation array names are equal.
        """
        return self._annot.keys() == item._annot.keys()
    
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
        elif attr in self._annot:
            return self._annot[attr]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )
        
    def __setattr__(self, attr, value):
        """
        If the attribute is an annotation, the `value` is saved to the
        annotation in the dictionary.
        Exposes coordinates.
        `value` must have same length as `array_length()`.
        """
        if attr == "coord":
            if not isinstance(value, np.ndarray):
                raise TypeError("Value must be ndarray of floats")
            if value.shape[-2] != self._array_length:
                raise ValueError(
                    f"Expected array length {self._array_length}, "
                    f"but got {len(value)}"
                )
            if value.shape[-1] != 3:
                raise TypeError("Expected 3 coordinates for each atom")
            self._coord = value
        if attr == "bonds":
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
        for name in self._annot.keys():
            attr.append(name)
    
    def __eq__(self, item):
        """
        See Also
        --------
        equal_annotations
        """
        if not self.equal_annotations(item):
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
        return concat
    
    def __copy_fill__(self, clone):
        super().__copy_fill__(clone)
        self._copy_annotations(clone)
        clone._coord = np.copy(self._coord)
    
    def _copy_annotations(self, clone):
        for name in self._annot:
            clone._annot[name] = np.copy(self._annot[name])
        if self._bonds is not None:
            clone._bonds = self._bonds.copy()
    

class Atom(object):
    """
    A representation of a single atom.
    
    The coordinates an annotations can be accessed directly.
    
    Parameters
    ----------
    coord: list or ndarray
        the x, y and z coordinates
    kwargs
        atom annotations as key value pair
    
    Attributes
    ----------
    {annot} : scalar
        Annotations for this atom.
    coord : ndarray, dtype=float
        ndarray containing the x, y and z coordinate of the atom. 
    
    Examples
    --------
    
    >>> atom = Atom([1,2,3], chain_id="A")
    >>> atom.atom_name = "CA"
    >>> print(atom.atom_name)
    CA
    >>> print(atom.coord)
    [1 2 3]
        
    """
    
    def __init__(self, coord, **kwargs):
        self._annot = {}
        if "kwargs" in kwargs:
            # kwargs are given directly as dictionary
            kwargs = kwargs["kwargs"]
        for name, annotation in kwargs.items():
            self._annot[name] = annotation
        coord = np.array(coord, dtype=float)
        # Check if coord contains x,y and z coordinates
        if coord.shape != (3,):
            raise ValueError("Position must be ndarray with shape (3,)")
        self.coord = coord
        
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
        return "{:3} {:3} {:5d} {:3} {:6} {:2}     {:8.3f} {:8.3f} {:8.3f}" \
               .format(hetero, self.chain_id, self.res_id, self.res_name,
                       self.atom_name, self.element,
                       self.coord[0], self.coord[1], self.coord[2])
    
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

    
class AtomArray(_AtomArrayBase):
    """
    An array representation of a model consisting of multiple atoms.
    
    An `AtomArray` can be seen as a list of `Atom` instances.
    Instead of using directly a list, this class uses an `NumPy`
    `ndarray` for each annotation category and the coordinates. These
    coordinates can be accessed directly via the `coord` attribute. The
    annotations are accessed either via the category as attribute name
    or the `get_annotation()`, `set_annotation()` method. Usage of
    custom annotations is achieved via `add_annotation()` or
    `set_annotation()`.
    
    In order to get an an subarray of an `AtomArray`, `NumPy` style
    indexing is used. This includes slices, boolean arrays,
    index arrays and even *Ellipsis* notation. Using a single integer as
    index returns a single `Atom` instance.
    
    Inserting or appending an `AtomArray` into another `AtomArray` is
    done with the '+' operator. Only the annotation categories, which
    are existing in both arrays, are transferred to the new array.

    Optionally, an `AtomArray` can store chemical bond information via
    a `BondList` object. It can be accessed using the `bonds` attribute.
    If no bond information is available, `bonds` is *None*.
    Consequently the bond information can be removed from the
    `AtomArray`, by setting `bonds` to *None*.
    When indexing the `AtomArray` the atom indics in the associated
    `BondList` are updated as well, hence the indices in the `BondList`
    will always point to the same atoms.
    If two `AtomArray` instances are concatenated, the resulting
    `AtomArray` will contain the merged `BondList` if at least one of
    the operands contains bond information.
    
    Parameters
    ----------
    length : int
        The fixed amount of atoms in the array.
    
    Attributes
    ----------
    {annot} : ndarray
        Mutliple n-length annotation arrays.
    coord : ndarray, dtype=float, shape=(n,3)
        ndarray containing the x, y and z coordinate of the
        atoms.
    bonds: BondList or None
        A `BondList`, specifying the indices of atoms
        that form a chemical bond.
    
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
    [[1 2 3]
     [2 3 4]
     [3 4 5]]
    
    `NumPy` style filtering:
    
    >>> atom_array = atom_array[atom_array.chain_id == "A"]
    >>> print(atom_array.array_length())
    2
        
    Inserting an atom array:
        
    >>> insert = array(Atom([7,8,9], chain_id="C"))
    atom_array = atom_array[0:1] + insert + atom_array[1:2]
    >>> print(atom_array.chain_id)
    ['A' 'C' 'A']
    """
    
    def __init__(self, length):
        super().__init__(length)
        if length is None:
            self._coord = None
        else:
            self._coord = np.full((length, 3), np.nan, dtype=float)
    
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
            If `index` is an integer an `Atom` instance,
            otherwise an `AtomArray` with reduced length is returned.
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
        Check if the array equals another `AtomArray`
        
        Parameters
        ----------
        item : object
            Object to campare the array with.
        
        Returns
        -------
        equal : bool
            True, if `item` is an `AtomArray`
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
    A collection of multiple `AtomArray` instances, where each atom
    array has equal annotation arrays.
    
    Effectively, this means that each atom is occuring in every array in
    the stack at differing coordinates. This situation arises e.g. in
    NMR-elucidated or simulated structures. Since the annotations are
    equal for each array the annotaion arrays are 1-D, while the
    coordinate array is 3-D (m x n x 3).
    
    Indexing works similar to `AtomArray`, with the difference, that two
    index dimensions are possible: The first index dimension specifies
    the array(s), the second index dimension specifies the atoms in each
    array (same as the index in `AtomArray`). Using a single integer as
    first dimension index returns a single `AtomArray` instance.
    
    Concatenation of atoms for each array in the stack is done using the
    '+' operator. For addition of atom arrays onto the stack use the
    `stack()` method.
    
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
        A `BondList`, specifying the indices of atoms
        that form a chemical bond.
    
    See also
    --------
    AtomArray
    
    Examples
    --------
    Creating an atom array stack from two arrays:
    
    >>> atom1 = Atom([1,2,3], chain_id="A")
    >>> atom1 = Atom([2,3,4], chain_id="A")
    >>> atom1 = Atom([3,4,5], chain_id="B")
    >>> atom_array1 = array(atom_array)
    >>> print(atom_array1.coord)
    [[1 2 3]
     [2 3 4]
     [3 4 5]]
    >>> atom_array2 = atom_array1.copy()
    >>> atom_array2.coord += 3
    >>> print(atom_array2.coord)
    [[4 5 6]
     [5 6 7]
     [6 7 8]]
    >>> array_stack = stack([atom_array1, atom_array2])
    >>> print(array_stack.coord)
    [[[1 2 3]
      [2 3 4]
      [3 4 5]]
    <BLANKLINE>
     [[4 5 6]
      [5 6 7]
      [6 7 8]]]
    """
    
    def __init__(self, depth, length):
        super().__init__(length)
        if depth == None or length == None:
            self._coord = None
        else:
            self._coord = np.zeros((depth, length, 3), dtype=float)
    
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
            All index types `NumPy` accepts are valid.
        
        Returns
        -------
        sub_array : AtomArray or AtomArrayStack
            If `index` is an integer an `AtomArray` instance,
            otherwise an `AtomArrayStack` with reduced depth and length
            is returned. In case the index is a tuple(int, int) an
            `Atom` instance is returned.  
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
                # Prevent reduction in dimensionality in second dimension
                if isinstance(index[1], numbers.Integral):
                    # Prevent reduction in dimensionality
                    # in second dimension
                    new_stack = self._subarray(slice(index[1], index[1]+1))
                else:
                    new_stack = self._subarray(index[1])
                if index[0] is not Ellipsis:
                    new_stack._coord = new_stack._coord[index[0]]
                return new_stack
        else:
            new_stack = AtomArrayStack(depth=0, length=self.array_length())
            self._copy_annotations(new_stack)
            new_stack._coord = self._coord[index]
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
        if not super(AtomArray, array).__eq__(array):
            raise ValueError(
                "The stack and the array have unequal annotations"
            )
        if isinstance(index, numbers.Integral):
            self._coord[index] = array._coord
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
        The depth of the stack.
        
        Returns
        -------
        depth : int
            depth of the array.
        """
        # length is determined by length of coord attribute
        return self._coord.shape[0]
    
    def __eq__(self, item):
        """
        Check if the array equals another `AtomArray`
        
        Parameters
        ----------
        item : object
            Object to campare the array with.
        
        Returns
        -------
        equal : bool
            True, if `item` is an `AtomArray`
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
        
        `AtomArray` strings eparated by blank lines
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
    Create an `AtomArray` from a list of `Atom`.
    
    Parameters
    ----------
    atoms : iterable(Atom)
        The atoms to be combined in an array.
    
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
    >>> print(atom_array.array_length())
    3
    """
    # Check if all atoms have the same annotation names
    # Equality check requires sorting
    names = sorted(atoms[0]._annot.keys())
    for atom in atoms:
        if sorted(atom._annot.keys()) != names:
            raise ValueError(
                "The atoms do not share the same annotation categories"
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
    Create an `AtomArrayStack` from a list of `AtomArray`.
    
    All atom arrays must have equal annotation arrays.
    
    Parameters
    ----------
    arrays : iterable object, type=AtomArray
        The atom arrays to be combined in a stack.
    
    Returns
    -------
    stack : AtomArrayStack
        The stacked atom arrays.
    
    Examples
    --------
    Creating an atom array stack from two arrays:
    
    >>> atom1 = Atom([1,2,3], chain_id="A")
    >>> atom1 = Atom([2,3,4], chain_id="A")
    >>> atom1 = Atom([3,4,5], chain_id="B")
    >>> atom_array1 = array(atom_array)
    >>> print(atom_array1.coord)
    [[1 2 3]
     [2 3 4]
     [3 4 5]]
    >>> atom_array2 = atom_array1.copy()
    >>> atom_array2.coord += 3
    >>> print(atom_array2.coord)
    [[4 5 6]
     [5 6 7]
     [6 7 8]]
    >>> array_stack = stack([atom_array1, atom_array2])
    >>> print(array_stack.coord)
    [[[1 2 3]
      [2 3 4]
      [3 4 5]]
    <BLANKLINE>
    [[4 5 6]
     [5 6 7]
     [6 7 8]]]
    """
    array_count = 0
    for array in arrays:
        array_count += 1
        # Check if all arrays share equal annotations
        if not array.equal_annotations(arrays[0]):
            raise ValueError("The atom arrays have unequal annotations")
    array_stack = AtomArrayStack(array_count, arrays[0].array_length())
    for name, annotation in arrays[0]._annot.items():
        array_stack._annot[name] = annotation
    coord_list = [array._coord for array in arrays] 
    array_stack._coord = np.stack(coord_list, axis=0)
    # Take bond list from first array
    array_stack._bonds = arrays[0]._bonds
    return array_stack


def coord(item):
    """
    Get the atom coordinates of the given array.
    
    This may be directly and `Atom`, `AtomArray` or `AtomArrayStack` or
    alternatively an (n x 3) or (m x n x 3) `ndarray`
    containing the coordinates.
    
    Parameters
    ----------
    item : `Atom`, `AtomArray` or `AtomArrayStack` or ndarray
        Takes the coord attribute, if `item` is `Atom`, `AtomArray` or
        `AtomArrayStack`, or directly returns a given `ndarray`.
    
    Returns
    -------
    coord : ndarray
        Atom coordinates.
    """

    if type(item) in (Atom, AtomArray, AtomArrayStack):
        return item.coord
    elif isinstance(item, np.ndarray):
        return item.astype(float, copy=False)
    else:
        return np.array(item, dtype=float)
