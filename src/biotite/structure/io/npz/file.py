# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["NpzFile"]

import numpy as np
from ...atoms import Atom, AtomArray, AtomArrayStack
from ....file import File


class NpzFile(File):
    """
    This class represents a NPZ file, the preferable format for
    Biotite internal structure storage. 
    
    Internally the this class writes/reads all attribute arrays of an
    `AtomArray` or `AtomArrayStack` using the `NumPy` `save()`/`load()`
    method. This format offers the fastest I/O operations and completely
    preserves the content all atom annotation arrays.
    
    Examples
    --------
    Load a \*.npz file, modify the structure and save the new
    structure into a new file:
    
    >>> file = NpzFile()
    >>> file.read("1l2y.npz")
    >>> array_stack = file.get_structure()
    >>> array_stack_mod = rotate(array_stack, [1,2,3])
    >>> file = NpzFile()
    >>> file.set_structure(array_stack_mod)
    >>> file.write("1l2y_mod.npz")
    
    """
    
    def __init__(self):
        self._data_dict = None
    
    def __copy_fill__(self, clone):
        super().__copy_fill__(clone)
        if self._data_dict is not None:
            for key, value in self._data_dict.items():
                clone._data_dict[key] = np.copy(value)
    
    def read(self, file):
        """
        Parse a NPZ file.
        
        Parameters
        ----------
        file : file-like object or str
            The file to be read.
            Alternatively, a file path can be supplied.
        """
        def _read(file):
            nonlocal self
            self._data_dict = dict(np.load(file, allow_pickle=False))
        
        if isinstance(file, str):
            with open(file, "rb") as f:
                _read(f)
        else:
            _read(file)
                
    def write(self, file):
        """
        Write a NPZ file.
        
        Parameters
        ----------
        file : file-like object or str
            The file to be read.
            Alternatively, a file path can be supplied.
        """
        def _write(file):
            nonlocal self
            np.savez(file, **self._data_dict)

        if isinstance(file, str):
            with open(file, "wb") as f:
                _write(f)
        else:
            _write(file)
    
    def get_structure(self):
        """
        Get an `AtomArray` or `AtomArrayStack` from the file.
        
        If this method returns an array or stack depends on which type
        of object was used when the file was written.
        
        Returns
        -------
        array : AtomArray or AtomArrayStack
            The array or stack in the content of this file
        """
        if self._data_dict is None:
            raise ValueError("The structure of this file "
                             "has not been loaded or set yet")
        coord = self._data_dict["coord"]
        # The type of the structure is determined by the dimensionality
        # of the 'coord' field
        if len(coord.shape) == 3:
            array = AtomArrayStack(coord.shape[0], coord.shape[1])
        else:
            array = AtomArray(coord.shape[0])
        array.coord = coord
        for key, value in self._data_dict.items():
            if key != "coord":
                array.set_annotation(key, value)
        return array
        
    def set_structure(self, array):
        """
        Set the `AtomArray` or `AtomArrayStack` for the file.
        
        Parameters
        ----------
        array : AtomArray or AtomArrayStack
            The array or stack to be saved into this file.
        """
        self._data_dict = {}
        self._data_dict["coord"] = np.copy(array.coord)
        for annot in array.get_annotation_categories():
            self._data_dict[annot] = np.copy(array.get_annotation(annot))