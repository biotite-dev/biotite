# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from ...atoms import Atom, AtomArray, AtomArrayStack

class NpzFile(object):
    
    def __init__(self):
        self._data_dict = None
    
    def read(self, file_name):
        self._data_dict = dict(np.load(file_name, allow_pickle=False))
                
    def write(self, file_name):
        np.savez(file_name, **self._data_dict)
    
    def copy(self):
        npz_file = NpzFile()
        if self._data_dict is not None:
            for key, value in self._data_dict.items():
                npz_file[key] = np.copy(value)
    
    def get_structure(self):
        if self._data_dict is None:
            raise ValueError("The structure of this file has not been loaded or set yet")
        coord = self._data_dict["coord"]
        if len(coord.shape) == 3:
            array = AtomArrayStack()
        else:
            array = AtomArray()
        array.coord = coord
        for key, value in self._data_dict.items():
            if key != "coord":
                array.add_annotation(key)
                array.set_annotation(key, value)
        return array
        
    def set_structure(self, array):
        self._data_dict = {}
        self._data_dict["coord"] = np.copy(array.coord)
        for annot in array.get_annotation_categories():
            self._data_dict[annot] = np.copy(array.get_annotation(annot))