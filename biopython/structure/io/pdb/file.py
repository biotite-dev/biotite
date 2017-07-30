# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from ...atoms import Atom, AtomArray, AtomArrayStack

_atom_records = {"hetero"    : (0,  6),
                 "atom_id"   : (6,  11),
                 "atom_name" : (12, 16),
                 "alt_loc"   : (16, ),
                 "res_name"  : (17, 20),
                 "chain_id"  : (21, ),
                 "res_id"    : (22, 26),
                 "ins_code"  : (26, ),
                 "coord_x"   : (30, 38),
                 "coord_y"   : (38, 46),
                 "coord_z"   : (46, 54),
                 "occupancy" : (54, 60),
                 "temp_f"    : (60, 66),
                 "element"   : (76, 78),
                 "charge"    : (78, 80),}

class PDBFile(object):
    
    def __init__(self):
        self._lines = []
    
    def read(self, file_name):
        raise NotImplementedError()
                
    def write(self, file_name):
        with open(file_name, "w") as f:
            f.writelines([line+"\n" for line in self._lines])
    
    def copy(self):
        pdb_file = PDBFile()
        pdb_file._lines = copy.deepcopy(self._lines)
        pdb_file._categories = copy.deepcopy(self._categories)
    
    def get_structure(self):
        raise NotImplementedError()
        
    def set_structure(self, array):
        self._lines = [" " * 79] * len(array)
        atom_id = np.arange(1, len(array)+1)
        hetero = ["ATOM" if e == False else "HETATM" for e in array.hetero]
        try:
            charge = [ "+"+str(np.abs(charge)) if charge > 0 else
                      ("-"+str(np.abs(charge)) if charge < 0 else
                       "")
                      for charge in array.get_annotation("charge")]
        except ValueError:
            # In case charge annotation is not existing
            charge = [""] * len(array)
        for i in range(len(array)):
            self._lines[i] = ("{:6}".format(hetero[i]) + 
                              "{:>5d}".format(atom_id[i]) +
                              " " +
                              "{:4}".format(array.atom_name[i]) +
                              " " +
                              "{:3}".format(array.res_name[i]) +
                              " " +
                              "{:1}".format(array.chain_id[i]) +
                              "{:>4d}".format(array.res_id[i]) +
                              (" " * 4) +
                              "{:>8.3f}".format(array.coord[i,0]) +
                              "{:>8.3f}".format(array.coord[i,1]) +
                              "{:>8.3f}".format(array.coord[i,2]) +
                              (" " * 22) +
                              "{:2}".format(array.element[i]) +
                              "{:2}".format(charge[i])
                             )
            