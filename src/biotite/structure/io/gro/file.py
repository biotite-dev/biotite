# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Daniel Bauer"
__all__ = ["GROFile"]

import numpy as np
from ...atoms import AtomArray, AtomArrayStack
from ....file import TextFile
from ...error import BadStructureError
import copy
import re

_atom_records = {"res_id"    : (0, 5),
                 "res_name"  : (5,10),
                 "atom_name" : (10,15),
                 "atom_id"   : (15,20),
                 "coord_x"   : (20, 28),
                 "coord_y"   : (28, 36),
                 "coord_z"   : (36, 44),
                 "v_x"       : (44, 52),
                 "v_y"       : (52, 60),
                 "v_z"       : (60, 68)}


class GROFile(TextFile):
    """
    This class represents a GRO file.

    This class only provides support for reading/writing the pure atom
    information

    Examples
    --------
    Load a `\*.gro` file, modify the structure and save the new
    structure into a new file:
    
    >>> file = GROFile()
    >>> file.read("1l2y.gro")
    >>> array_stack = file.get_structure()
    >>> array_stack_mod = rotate(array_stack, [1,2,3])
    >>> file = GROFile()
    >>> file.set_structure(array_stack_mod)
    >>> file.write("1l2y_mod.gro")
    
    """
    
    def get_structure(self, insertion_code=[], altloc=[],
                      model=None, extra_fields=[]):
        """
        Get an `AtomArray` or `AtomArrayStack` from the GRO file.
        
        Parameters
        ----------
        insertion_code : list of tuple, optional
            In case the structure contains insertion codes, those can be
            specified here: Each tuple consists of an integer, specifying
            the residue ID, and a letter, specifying the insertion code.
            By default no insertions are used.
        altloc : list of tuple, optional
            In case the structure contains *altloc* entries, those can be
            specified here: Each tuple consists of an integer, specifying
            the residue ID, and a letter, specifying the *altloc* ID.
            By default the location with the *altloc* ID "A" is used.
        model : int, optional
            If this parameter is given, the function will return an
            `AtomArray` from the atoms corresponding to the given model ID.
            If this parameter is omitted, an `AtomArrayStack` containing all
            models will be returned, even if the structure contains only one
            model.
        extra_fields : list of str, optional
            The strings in the list are optional annotation categories
            that should be stored in the output array or stack.
            There are 4 optional annotation identifiers:
            'atom_id', 'b_factor', 'occupancy' and 'charge'.
        
        Returns
        -------
        array : AtomArray or AtomArrayStack
            The return type depends on the `model` parameter.
        """

        def is_int(line):
            try:
                int(line)
                return True
            except ValueError:
                return False

        # Line indices where a new model starts
        model_start_i = np.array([i for i in range(len(self._lines))
                                                     if is_int(self._lines[i])],
                                                    dtype=int)

        # Number of atoms in each model
        model_atom_counts = np.array([int(self._lines[i]) for i in model_start_i])

        # Helper function to get the indeces of all atoms for a model
        def get_atom_line_i(model_start_i, model_atom_counts):
            return np.arange(model_start_i+1, model_start_i+1+model_atom_counts)

        if model is None:
            # Check if all models have the same length
            if np.all(model_atom_counts != model_atom_counts[0]):
                raise BadStructureError("The models in the file have unequal "
                                        "amount of atoms, give an explicit "
                                        "model instead")
            depth = len(model_start_i)
            length = model_atom_counts[0]
            array = AtomArrayStack(depth, length)

            # Line indices for annotation determination is determined from model 1,
            annot_i = get_atom_line_i(model_start_i[0], length)
        else:
            if model > len(model_start_i):
                raise ValueError("Requested model {:d} is larger than the "
                           "amount of models ({:d})"
                           .format(model, len(model_start_i)))

            length = model_atom_counts[model+1]
            array = AtomArray(length)

            annot_i = get_atom_line_i(model_start_i[model+1], length)
            coord_i = get_atom_line_i(model_start_i[model+1], length)

        # Add optional annotation arrays
        if "atom_id" in extra_fields:
            array.add_annotation("atom_id", dtype=int)

        # Fill in annotation
        def guess_element(atom_name):
            if re.match(r'[123]H', atom_name):
                return 'H'
            else:
                return atom_name[0]


        # i is index in array, j is line index
        for i, line_i in enumerate(annot_i):
            line = self._lines[line_i]
            array.res_id[i] = int(line[0:5])
            array.res_name[i] = line[5:10].strip()
            array.atom_name[i] = line[10:15].strip()
            array.element[i] = guess_element(line[10:15].strip())

        if extra_fields:
            for i, line_i in enumerate(annot_i):
                line = self._lines[line_i]
                if "atom_id" in extra_fields:
                    array.atom_id[i] = int(line[15:20].strip())

        # Fill in coordinates
        if isinstance(array, AtomArray):
            for i, line_i in enumerate(coord_i):
                line = self._lines[line_i]
                array.coord[i,0] = float(line[20:28])
                array.coord[i,1] = float(line[28:36])
                array.coord[i,2] = float(line[36:44])
        elif isinstance(array, AtomArrayStack):
            for m in range(len(model_start_i)):
                for atom_i, line_i in zip(np.arange(0, model_atom_counts[0]), get_atom_line_i(model_start_i[m], model_atom_counts[m])):
                    line = self._lines[i]
                    array.coord[m,i,0] = float(line[20:28])
                    array.coord[m,i,1] = float(line[28:36])
                    array.coord[m,i,2] = float(line[36:44])

        return array

            
    def set_structure(self, array):
        """
        Set the `AtomArray` or `AtomArrayStack` for the file.
        
        This makes also use of the optional annotation arrays
        'atom_id', 'b_factor', 'occupancy' and 'charge'.
        
        Parameters
        ----------
        array : AtomArray or AtomArrayStack
            The array or stack to be saved into this file. If a stack
            is given, each array in the stack is saved as separate
            model.
        """
        # Save list of annotation categories for checks,
        # if an optional category exists
        annot_categories = array.get_annotation_categories()
        hetero = ["ATOM" if e == False else "HETATM" for e in array.hetero]
        if "atom_id" in annot_categories:
            atom_id = array.atom_id
        else:
            atom_id = np.arange(1, array.array_length()+1)
        if "b_factor" in annot_categories:
            b_factor = array.b_factor
        else:
            b_factor = np.zeros(array.array_length())
        if "occupancy" in annot_categories:
            occupancy = array.occupancy
        else:
            occupancy = np.ones(array.array_length())
        if "charge" in annot_categories:
            charge = [ "+"+str(np.abs(charge)) if charge > 0 else
                      ("-"+str(np.abs(charge)) if charge < 0 else
                       "")
                      for charge in array.get_annotation("charge")]
        else:
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
                                  "{:>6.2f}".format(occupancy[i]) +
                                  "{:>6.3f}".format(b_factor[i]) +
                                  (" " * 10) + 
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
                                 (" " * 28) +
                                 "{:>6.2f}".format(occupancy[i]) +
                                 "{:>6.3f}".format(b_factor[i]) +
                                 (" " * 10) +
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
                
            