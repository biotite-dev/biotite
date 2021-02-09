# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.gro"
__author__ = "Daniel Bauer, Patrick Kunzmann"
__all__ = ["GROFile"]

import numpy as np
from ...atoms import AtomArray, AtomArrayStack
from ...box import is_orthogonal
from ....file import TextFile, InvalidFileError
from ..general import _guess_element as guess_element
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
    r"""
    This class represents a GRO file.

    This class only provides support for reading/writing the pure atom
    information

    Examples
    --------
    Load a `\\*.gro` file, modify the structure and save the new
    structure into a new file:
    
    >>> import os.path
    >>> file = GROFile.read(os.path.join(path_to_structures, "1l2y.gro"))
    >>> array_stack = file.get_structure()
    >>> array_stack_mod = rotate(array_stack, [1,2,3])
    >>> file = GROFile()
    >>> file.set_structure(array_stack_mod)
    >>> file.write(os.path.join(path_to_directory, "1l2y_mod.gro"))
    
    """
    def get_model_count(self):
        """
        Get the number of models contained in this GRO file.

        Returns
        -------
        model_count : int
            The number of models.
        """
        model_count = 0
        for line in self.lines:
            if _is_int(line):
                model_count += 1
        return model_count


    def get_structure(self, model=None):
        """
        Get an :class:`AtomArray` or :class:`AtomArrayStack` from the
        GRO file.
        
        Parameters
        ----------
        model : int, optional
            If this parameter is given, the function will return an
            :class:`AtomArray` from the atoms corresponding to the given
            model number (starting at 1).
            Negative values are used to index models starting from the
            last model insted of the first model.
            If this parameter is omitted, an :class:`AtomArrayStack`
            containing all models will be returned, even if the
            structure contains only one model.
        
        Returns
        -------
        array : AtomArray or AtomArrayStack
            The return type depends on the `model` parameter.
        """
        
        def get_atom_line_i(model_start_i, model_atom_counts):
            """
            Helper function to get the indices of all atoms for a model
            """
            return np.arange(
                model_start_i+1, model_start_i+1+model_atom_counts
            )
        
        def set_box_dimen(box_param):
            """
            Helper function to create the box vectors from the values
            in the GRO file

            Parameters
            ----------
            box_param : list of float
                The box dimensions in the GRO file.
            
            Returns
            -------
            box_vectors : ndarray, dtype=float, shape=(3,3)
                The atom array compatible box vectors.
            """
            if not any(box_param):
                return None
            if len(box_param) == 3:
                x, y, z = box_param
                return np.array([[x,0,0], [0,y,0], [0,0,z]], dtype=float)
            elif len(box_param) == 9:
                x1, y2, z3, x2, x3, y1, y3, z1, z2 = box_param
                return np.array(
                    [[x1,x2,x3], [y1,y2,y3], [z1,z2,z3]], dtype=float
                )
            else:
                raise InvalidFileError(
                    f"Invalid amount of box parameters: {len(box_param)}"
                )

        # Line indices where a new model starts
        model_start_i = np.array([i for i in range(len(self.lines))
                                  if _is_int(self.lines[i])],
                                 dtype=int)

        # Number of atoms in each model
        model_atom_counts = np.array(
            [int(self.lines[i]) for i in model_start_i]
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
            if model == 0:
                raise ValueError("The model index must not be 0")
            # Negative models mean index starting from last model
            model = len(model_start_i) + model + 1 if model < 0 else model
            if model > len(model_start_i):
                raise ValueError(
                    f"The file has {len(model_start_i)} models, "
                    f"the given model {model} does not exist"
                )

            length = model_atom_counts[model-1]
            array = AtomArray(length)

            annot_i = get_atom_line_i(model_start_i[model-1], length)

        # Replace empty strings for elements with guessed types
        # i is index in array, line_i is line index
        for i, line_i in enumerate(annot_i):
            line = self.lines[line_i]
            array.res_id[i] = int(line[0:5])
            array.res_name[i] = line[5:10].strip()
            array.atom_name[i] = line[10:15].strip()
            array.element[i] = guess_element(line[10:15].strip())

        # Fill in coordinates and boxes
        if isinstance(array, AtomArray):
            atom_i = annot_i
            for i, line_i in enumerate(atom_i):
                line = self.lines[line_i]
                # gro files use nm instead of A
                array.coord[i,0] = float(line[20:28])*10
                array.coord[i,1] = float(line[28:36])*10
                array.coord[i,2] = float(line[36:44])*10
            # Box is stored in last line (after coordinates)
            box_i = atom_i[-1] + 1
            box_param = [float(e)*10 for e in self.lines[box_i].split()]
            array.box = set_box_dimen(box_param)
        
        elif isinstance(array, AtomArrayStack):
            for m in range(len(model_start_i)):
                atom_i = get_atom_line_i(
                    model_start_i[m], model_atom_counts[m]
                )
                for i, line_i in enumerate(atom_i):
                    line = self.lines[line_i]
                    array.coord[m,i,0] = float(line[20:28])*10
                    array.coord[m,i,1] = float(line[28:36])*10
                    array.coord[m,i,2] = float(line[36:44])*10
                # Box is stored in last line (after coordinates)
                box_i = atom_i[-1] + 1
                box_param = [float(e)*10 for e in self.lines[box_i].split()]
                box = set_box_dimen(box_param)
                # Create a box in the stack if not already existing
                # and the box is not a dummy
                if box is not None:
                    if array.box is None: 
                        array.box = np.zeros((array.stack_depth(), 3, 3))
                    array.box[m] = box
                    
        return array

            
    def set_structure(self, array):
        """
        Set the :class:`AtomArray` or :class:`AtomArrayStack` for the
        file.
        
        Parameters
        ----------
        array : AtomArray or AtomArrayStack
            The array or stack to be saved into this file. If a stack
            is given, each array in the stack is saved as separate
            model.
        """
        def get_box_dimen(array):
            """
            GRO files have the box dimensions as last line for each
            model.
            In case, the `box` attribute of the atom array is
            `None`, we simply use the min and max coordinates in xyz
            to get the correct size

            Parameters
            ----------
            array : AtomArray
                The atom array to get the box dimensions from.
            
            Returns
            -------
            box : str
                The box, properly formatted for GRO files.
            """
            if array.box is None:
                coord = array.coord
                bx, by, bz = (coord.max(axis=0) - coord.min(axis=0)) / 10
                return f"{bx:>8.3f} {by:>8.3f} {bz:>8.3f}"
            else:
                box = array.box
                if is_orthogonal(box):
                    bx, by, bz = np.diag(box) / 10
                    return f"{bx:>9.5f} {by:>9.5f} {bz:>9.5f}"
                else:
                    box = box / 10
                    box_elements = (
                        box[0,0], box[1,1], box[2,2],
                        box[0,1], box[0,2],
                        box[1,0], box[1,2],
                        box[2,0], box[2,1],
                    )
                    return " ".join([f"{e:>9.5f}" for e in box_elements])
        
        if "atom_id" in array.get_annotation_categories():
            atom_id = array.atom_id
        else:
            atom_id = np.arange(1, array.array_length() + 1)
        # Atom IDs are supported up to 99999,
        # but negative IDs are also possible
        gro_atom_id = np.where(
            atom_id > 0,
            ((atom_id - 1) % 99999) + 1,
            atom_id
        )
        # Residue IDs are supported up to 9999,
        # but negative IDs are also possible
        gro_res_id = np.where(
            array.res_id > 0,
            ((array.res_id - 1) % 99999) + 1,
            array.res_id
        )

        if isinstance(array, AtomArray):
            self.lines = [None] * (array.array_length() + 3)

            # Write header lines
            self.lines[0] = f"Generated by Biotite at {datetime.now()}"
            self.lines[1] = str(array.array_length())

            # Write atom lines
            fmt = "{:>5d}{:5s}{:>5s}{:>5d}{:>8.3f}{:>8.3f}{:>8.3f}"
            for i in range(array.array_length()):
                # gro format is in nm -> multiply coords by 10
                self.lines[i+2] = fmt.format(
                    gro_res_id[i], array.res_name[i], array.atom_name[i],
                    gro_atom_id[i], array.coord[i,0]/10, array.coord[i,1]/10,
                    array.coord[i,2]/10
                )
            # Write box lines
            self.lines[-1] = get_box_dimen(array)
        elif isinstance(array, AtomArrayStack):
            self.lines = []
            # The entire information, but the coordinates,
            # is equal for each model
            # Therefore template lines are created
            # which are afterwards applied for each model
            templines = [None] * array.array_length()
            fmt = '{:>5d}{:5s}{:>5s}{:5d}'
            for i in range(array.array_length()):
                templines[i] = fmt.format(gro_res_id[i], array.res_name[i],
                                           array.atom_name[i], gro_atom_id[i])

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
                self.lines.append(get_box_dimen(array[i]))
        else:
            raise TypeError("An atom array or stack must be provided")
        # Add terminal newline, since PyMOL requires it
        self.lines.append("")


def _is_int(string):
    """
    Return ``True`, if the string can be parsed to an int.
    """
    try:
        int(string)
        return True
    except ValueError:
        return False