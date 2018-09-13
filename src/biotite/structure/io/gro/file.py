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
from datetime import datetime

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
    
    def get_structure(self, model=None):
        """
        Get an `AtomArray` or `AtomArrayStack` from the GRO file.
        
        Parameters
        ----------
        model : int, optional
            If this parameter is given, the function will return an
            `AtomArray` from the atoms corresponding to the given model
            ID.
            If this parameter is omitted, an `AtomArrayStack` containing
            all models will be returned, even if the structure contains
            only one model.
        
        Returns
        -------
        array : AtomArray or AtomArrayStack
            The return type depends on the `model` parameter.
        """

        def is_int(line):
            """
            helper function: returns true
            if the string can be parsed to an int
            """
            try:
                int(line)
                return True
            except ValueError:
                return False

        # Line indices where a new model starts
        model_start_i = np.array([i for i in range(len(self.lines))
                                  if is_int(self.lines[i])],
                                 dtype=int)

        # Number of atoms in each model
        model_atom_counts = np.array(
            [int(self.lines[i]) for i in model_start_i]
        )

        # Helper function to get the indeces of all atoms for a model
        def get_atom_line_i(model_start_i, model_atom_counts):
            return np.arange(
                model_start_i+1, model_start_i+1+model_atom_counts
            )

        if model is None:
            # Check if all models have the same length
            if np.all(model_atom_counts != model_atom_counts[0]):
                raise BadStructureError("The models in the file have unequal "
                                        "amount of atoms, give an explicit "
                                        "model instead")
            depth = len(model_start_i)
            length = model_atom_counts[0]
            array = AtomArrayStack(depth, length)

            # Line indices for annotation determination is determined
            # from model 1
            annot_i = get_atom_line_i(model_start_i[0], length)
        else:
            if model > len(model_start_i):
                raise ValueError(
                    f"Requested model {model} is larger than the "
                    f"amount of models ({len(model_start_i)})"
                )

            length = model_atom_counts[model-1]
            array = AtomArray(length)

            annot_i = get_atom_line_i(model_start_i[model-1], length)
            coord_i = get_atom_line_i(model_start_i[model-1], length)

        # Fill in elements
        def guess_element(atom_name):
            if atom_name.startswith(("H", "1H", "2H", "3H")):
                return 'H'
            else:
                return atom_name[0]

        # i is index in array, line_i is line index
        for i, line_i in enumerate(annot_i):
            line = self.lines[line_i]
            array.res_id[i] = int(line[0:5])
            array.res_name[i] = line[5:10].strip()
            array.atom_name[i] = line[10:15].strip()
            array.element[i] = guess_element(line[10:15].strip())

        # Fill in coordinates
        if isinstance(array, AtomArray):
            for i, line_i in enumerate(coord_i):
                line = self.lines[line_i]
                # gro files use nm instead of A
                array.coord[i,0] = float(line[20:28])*10
                array.coord[i,1] = float(line[28:36])*10
                array.coord[i,2] = float(line[36:44])*10
        elif isinstance(array, AtomArrayStack):
            for m in range(len(model_start_i)):
                atom_i = np.arange(0, model_atom_counts[0])
                line_i = get_atom_line_i(model_start_i[m], model_atom_counts[m])
                for atom_i, line_i in zip(atom_i, line_i):
                    line = self.lines[line_i]
                    array.coord[m,atom_i,0] = float(line[20:28])*10
                    array.coord[m,atom_i,1] = float(line[28:36])*10
                    array.coord[m,atom_i,2] = float(line[36:44])*10

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

        def get_box_dimen(array):
            """
            GRO files have the box dimensions as last line for each model.
            Because we cannot properly detect the box shape, we simply use
            the min and max coordinates in xyz to get the correct size
            """
            return np.abs(array.coord.max(axis=0) - array.coord.min(axis=0))/10

        if isinstance(array, AtomArray):
            self.lines = [None] * (array.array_length() + 3)

            # Write header lines
            self.lines[0] = f"Generated by Biotite at {datetime.now()}"
            self.lines[1] = str(array.array_length())

            # Write atom lines
            fmt = '{:>5d}{:5s}{:>5s}{:>5d}{:>8.3f}{:>8.3f}{:>8.3f}'
            for i in range(array.array_length()):
                # gro format is in nm -> multiply coords by 10
                self.lines[i+2] = fmt.format(
                    array.res_id[i], array.res_name[i], array.atom_name[i],
                    atom_id[i], array.coord[i,0]/10, array.coord[i,1]/10,
                    array.coord[i,2]/10
                )
            self.lines[-1] = "{:>8.3f} {:>8.3f} {:>8.3f}" \
                              .format(*get_box_dimen(array))
        elif isinstance(array, AtomArrayStack):
            self.lines = []
            # The entire information, but the coordinates,
            # is equal for each model
            # Therefore template lines are created
            # which are afterwards applied for each model
            templines = [None] * array.array_length()
            fmt = '{:>5d}{:5s}{:>5s}{:5d}'
            for i in range(array.array_length()):
                templines[i] = fmt.format(array.res_id[i], array.res_name[i],
                                           array.atom_name[i], atom_id[i])

            for i in range(array.stack_depth()):
                self.lines.append(
                    f"Generated by Biotite at {datetime.now()}, model={i+1}"
                )
                self.lines.append(str(array.array_length()))

                # Fill in coordinates for each model
                modellines = copy.copy(templines)
                for j, line in enumerate(modellines):
                    # Insert coordinates
                    line = (line + "{:>8.3f}{:>8.3f}{:>8.3f}".format(
                                    array.coord[i,j,0]/10,
                                    array.coord[i,j,1]/10,
                                    array.coord[i,j,2]/10))
                    modellines[j] = line
                self.lines.extend(modellines)
                self.lines.append("{:>8.3f} {:>8.3f} {:>8.3f}"
                                   .format(*get_box_dimen(array[i])))

