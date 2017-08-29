# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from ...atoms import Atom, AtomArray, AtomArrayStack
from ....file import TextFile, register_suffix
from ...error import BadStructureError
import copy

__all__ = ["PDBFile"]


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


@register_suffix(["pdb"])
class PDBFile(TextFile):
    """
    This class represents a PDB file.
    
    The usage of PDBxFile is encouraged in favor of this class.
    
    This class only provides suppert reading/writing the pure atom
    information (*ATOM*, *HETATM, *MODEL* and *ENDMDL* records). *TER*
    records cannot be written.
    
    Usage of altlocs and insertion codes is not supported.
    
    See also
    --------
    PDBxFile
    
    Examples
    --------
    Load a `\*.pdb` file, modify the structure and save the new
    structure into a new file:
    
        >>> file = PDBFile()
        >>> file.read("1l2y.pdb")
        >>> array_stack = file.get_structure()
        >>> array_stack_mod = rotate(array_stack, [1,2,3])
        >>> file = PDBFile()
        >>> file.set_structure(array_stack_mod)
        >>> file.write("1l2y_mod.pdb")
    
    """
    
    def copy(self):
        pdb_file = PDBFile()
        pdb_file._lines = copy.deepcopy(self._lines)
        pdb_file._categories = copy.deepcopy(self._categories)
    
    def get_structure(self):
        """
        Get an `AtomArray` or `AtomArrayStack` from the PDB file.
        
        Returns
        -------
        array : AtomArray or AtomArrayStack
            A stack is returned, if this file contains multiple models,
            otherwise an array is returned.
        """
        # Line indices where a new model starts
        model_start_i = np.array([i for i in range(len(self._lines))
                                  if self._lines[i].startswith(("MODEL"))])
        # Line indices with ATOM or HETATM records
        # Filter out lines of altlocs and insertion codes
        atom_line_i = np.array([i for i in range(len(self._lines)) if
                                self._lines[i].startswith(("ATOM", "HETATM"))
                                # altloc
                                and self._lines[i][16] in [" ", "A"]
                                # inscode
                                and self._lines[i][26] == " "])
        
        model = 0
        if len(model_start_i) <= 1:
            array = AtomArray(len(atom_line_i))
            annot_i = atom_line_i
        else:
            if len(atom_line_i) % len(model_start_i):
                raise BadStructureError("Amount of ATOM/HETATM records is "
                                        "not multiple of model count")
            array = AtomArrayStack(len(model_start_i),
                                   len(atom_line_i) // len(model_start_i))
            annot_i = atom_line_i[atom_line_i < model_start_i[1]]
        # Fill in annotation
        # i is index in array, j is line index
        for i, j in enumerate(annot_i):
            line = self._lines[j]
            array.chain_id[i] = line[21].upper().strip()
            array.res_id[i] = int(line[22:26])
            array.res_name[i] = line[17:20].strip()
            array.hetero[i] = (False if line[0:4] == "ATOM" else True)
            array.atom_name[i] = line[12:16].strip()
            array.element[i] = line[76:78].strip()
        
        # Fill in coordinates
        if isinstance(array, AtomArray):
            for i, j in enumerate(atom_line_i):
                line = self._lines[j]
                array.coord[i,0] = float(line[30:38])
                array.coord[i,1] = float(line[38:46])
                array.coord[i,2] = float(line[46:54])
        
        elif isinstance(array, AtomArrayStack):
            m = 0
            i = 0
            for k in atom_line_i:
                if m < len(model_start_i)-1 and k > model_start_i[m+1]:
                    m += 1
                    i = 0
                line = self._lines[k]
                array.coord[m,i,0] = float(line[30:38])
                array.coord[m,i,1] = float(line[38:46])
                array.coord[m,i,2] = float(line[46:54])
                i += 1
                
        return array
        
    def set_structure(self, array):
        """
        Set the `AtomArray` or `AtomArrayStack` for the file.
        
        Parameters
        ----------
        array : AtomArray or AtomArrayStack
            The array or stack to be saved into this file. If a stack
            is given, each array in the stack is saved as separate
            model.
        """
        atom_id = np.arange(1, array.array_length()+1)
        hetero = ["ATOM" if e == False else "HETATM" for e in array.hetero]
        try:
            charge = [ "+"+str(np.abs(charge)) if charge > 0 else
                      ("-"+str(np.abs(charge)) if charge < 0 else
                       "")
                      for charge in array.get_annotation("charge")]
        except ValueError:
            # In case charge annotation is not existing
            charge = [""] * array.array_length()
        
        if isinstance(array, AtomArray):
            self._lines = [None] * array.array_length()
            for i in range(array.array_length()):
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
        
        elif isinstance(array, AtomArrayStack):
            self._lines = []
            # The entire information, but the coordinates,
            # is equal for each model
            # Therefore template lines are created
            # which are afterwards applied for each model
            temp_lines = [None] * array.array_length()
            for i in range(array.array_length()):
                temp_lines[i] = ("{:6}".format(hetero[i]) + 
                                 "{:>5d}".format(atom_id[i]) +
                                 " " +
                                 "{:4}".format(array.atom_name[i]) +
                                 " " +
                                 "{:3}".format(array.res_name[i]) +
                                 " " +
                                 "{:1}".format(array.chain_id[i]) +
                                 "{:>4d}".format(array.res_id[i]) +
                                 (" " * 50) + 
                                 "{:2}".format(array.element[i]) +
                                 "{:2}".format(charge[i])
                                )
            for i in range(array.stack_depth()):
                #Fill in coordinates for each model
                self._lines.append("{:5}{:>9d}".format("MODEL", i+1))
                model_lines = copy.copy(temp_lines)
                for j, line in enumerate(model_lines):
                    # Insert coordinates
                    line = (line[:30]
                            + "{:>8.3f}{:>8.3f}{:>8.3f}".format(
                                    array.coord[i,j,0],
                                    array.coord[i,j,1],
                                    array.coord[i,j,2])
                            + line[54:] )
                    model_lines[j] = line
                self._lines.extend(model_lines)
                self._lines.append("ENDMDL")
                
            