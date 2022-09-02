# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.xyz"
__author__ = "Benjamin E. Mayer"
__all__ = ["XYZFile"]

import numpy as np
from ...atoms import AtomArray, AtomArrayStack, Atom
import biotite.structure as struc
from ....file import TextFile


# Number of header lines
N_HEADER = 2


class XYZFile(TextFile):
    """
    This class represents a file in XYZ format, which is a very simple 
    file format where only a molecule_name in form of a string and the
    number of atoms is contained in the header. 
    Followed by as many lines containing atom element and 3D coordinates.
    
    References
    ----------
    
    .. footbibliography::
    
    Examples
    --------

    >>> from os.path import join
    >>> mol_file = XYZFile.read(join(path_to_structures, "molecules", "BENZ.xyz"))
    >>> atom_array = xyz_file.get_structure()
    >>> print(atom_array)    
            0             C         0.000    1.403    0.000
            0             H         0.000    2.490    0.000
            0             C        -1.215    0.701    0.000
            0             H        -2.157    1.245    0.000
            0             C        -1.215   -0.701    0.000
            0             H        -2.157   -1.245    0.000
            0             C         0.000   -1.403    0.000
            0             H         0.000   -2.490    0.000
            0             C         1.215   -0.701    0.000
            0             H         2.157   -1.245    0.000
            0             C         1.215    0.701    0.000
            0             H         2.157    1.245    0.000

    """
    
    def __init__(self):
        super().__init__()
        # empty header lines
        self.lines = [""] * N_HEADER
        self._mol_names = None
        self._atom_numbers = None
        self._model_start_inds = None
        self._structures = None
        self._model_start_inds = None
        
    def __update_start_lines(self):
        """            
        Internal function that is used to update the _model_start_inds
        private member variable where the indices of where a new 
        model within the xyz file read starts is stored.
        """

        # Line indices where a new model starts -> where number of atoms
        # is written down as number
    
        self._model_start_inds = np.array(
            [i for i in range(len(self.lines))
             if self.lines[i].strip().isdigit()],
             dtype=int
         )
#         
#        print(" ... pre processing ... ")                            
#        print("|model_start_inds| :: " + str(self._model_start_inds))
                     
        if self._model_start_inds.shape[0] > 1:             
            # if the mol name line only contains an integer
            # these lines will appear in model_start_inds 
            # if calculated as above, therfore we purge all 
            # indices that have a distance of 1 to their previous index
            # (this will not work with a file containing multiple models
            #  with solely one coordinate / atom per model )                                                     
            self._model_start_inds = self._model_start_inds[
                np.concatenate(
                    (
                        np.array([0],dtype=int), 
                        np.where(
                            self._model_start_inds[:-1]
                                -
                            self._model_start_inds[1:] 
                                != 
                            -1
                        )[0]+1
                    )
                )
            ] 
        elif self._model_start_inds.shape[0] == 2:                            
            self._model_start_inds = self._model_start_inds[:1]            
    
    def get_header(self, model=None):
    
        """
        Get the header from the XYZ file.
        
        Returns
        -------
        atom_number: int
            The number of atoms per model
        mol_name : str
            The name of the molecule or the names of the multiple models                
        """
        
        # Line indices where a new model starts -> where number of atoms
        # is written down as number 
        self.__update_start_lines()
                        
        # parse all atom_numbers into integers
        if self._atom_numbers is None:
            self._atom_numbers = [
                int(
                    self.lines[i].strip().strip(" ")
                ) for i in self._model_start_inds
            ]
          
        # parse all lines containing names             
        if self._mol_names is None:
            self._mol_names = [
                self.lines[i+1].strip() for i in self._model_start_inds
            ]        
        
        
        return self._atom_numbers, self._mol_names

    def __get_number_of_atoms(self):
        """
        This calculates the number of atoms from the previously read 
        file.
        """
        lines_parsed = [x.split() for x in self.lines]
        inds=np.array(
            [
                i for i in range(
                    len(lines_parsed)
                ) if len(lines_parsed[i]) != 4
            ]
        )       
        if inds.shape[0] == 2:
            return int(len(self.lines[2:]))
        else:             
            inds=inds[np.where(inds[1:]-inds[:-1]!=1)[0]]
            line_lengths=np.unique(inds[1:]-inds[:-1])
        
            if line_lengths.shape[0] > 1:
                msg  = "File contains different molecules."
                msg += "Currently only multiple models of the same molecule"
                msg += "are supported within one file"
                raise BadStructureError(msg)
        
            return np.unique(inds[1:]-inds[:-1])[0]        

    def set_header(self, mol_name):
        """
        Set the header for the XYZ file.
        As the header consist only out of the mol_name and the number 
        of atoms in the structure / structures this can only be 
        used after setting a structure. Since the second line 
        is calculated by counting the 
        
        Parameters
        ----------
        mol_name : str
            The name of the molecule.
  
        """
        
        if(len(self.lines) > 2):
            
            self.lines[1] = str(mol_name)    
            self.lines[0] = str(self.__get_number_of_atoms())
            
            self._mol_names = [mol_name]
            self._atom_numbers = [int(self.lines[0])]
            
            self.__update_start_lines()
        
        else:
            raise ValueError(
                    "Can not set header of an empty XYZFile"
                    "Use set_structure first, so that number of atoms"
                    "can be derived from set structure"
                )


    def get_structure(self, model=None):
        """
        Get an :class:`AtomArray` or :class:`AtomArrayStack` from the XYZ file.
        
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
        
        if len(self.lines) <= 2:         
            raise ValueError(
                        "Trying to get_structure from empty XYZFile"
            ) 
            
        atom_number, names = self.get_header()

        
        # set a default head if non present since
        # the number of lines will be calculated from the number
        # of atoms field in the file (counts how many lines with numbers
        # there are within the file which are not name lines)
        if len(names) == 0 or len(atom_number) == 0:
            self.set_header("[MOLNAME]")  
           
        self.__update_start_lines()
                
                            
        # parse all atom_numbers into integers
        if self._atom_numbers is None:
            self._atom_numbers = [
                int(
                    self.lines[i].strip().strip(" ")
                ) for i in self._model_start_inds
            ]
          
        # parse all lines containing names             
        if self._mol_names is None:
            self._mol_names = [
                self.lines[i+1].strip() for i in self._model_start_inds
            ]
            

        # parse all coordinates        
        if self._structures is None:                                   
                                 
            array_stack = []
        
            for i, ind in enumerate(self._model_start_inds):
                ind_end = ind+2 + self._atom_numbers[i]
                lines_cut = self.lines[ind:ind_end]  
                array = AtomArray(self._atom_numbers[i])
                
                if self._atom_numbers[i]+2 != len(lines_cut):
                    raise ValueError(
                        "Number of Atoms not matching with coordinate lines"
                        + ""
                        + " atom_number :: " + str(self._atom_numbers[i])
                        + ""
                        + ""
                        + " |lines_cut| :: " + str(len(lines_cut))
                        + " lines_cut   :: " + str(lines_cut)
                    )                         
                
                for j, line in enumerate(lines_cut[2:]):

                    line_parsed = [
                        x for x in line.strip().split(" ") if x!= ''
                    ]                    
                    x = float(line_parsed[1])
                    y = float(line_parsed[2])
                    z = float(line_parsed[3])
                    
                    if np.isnan(x) or np.isnan(y) or np.isnan(z):

                        raise ValueError(
                            "At least one of the coordinates is NaN"
                            ""
                            "(" + str(x) + "," + str(y) + "," +str(z) + ")"
                            ""
                        )                                                 
                        
                    atom = Atom([x,y,z])
                    atom.element = line_parsed[0]                
                    array[j] = atom
                                                            
                
                array_stack.append(array)
            self._structures = struc.stack(array_stack)
                                        
                
   
                
                    
        
        if model is None:            
            return self._structures
        else:
            return self._structures[model]                          
        
        

    def set_structure(self, atoms):
        """
        Set the :class:`AtomArray` :class:`AtomArrayStack` or for the file.
        Based upon the given type of the atoms parameter either a single 
        model or multiple model XYZFile will be contained within the 
        lines member variable of this XYZFile instance.
        
        Parameters
        ----------
        atoms : AtomArray, AtomArrayStack
            The array to be saved into this file.
        """
        
        if isinstance(atoms, AtomArray):
        
            n_atoms = atoms.shape[0]
            self.lines[0] = str(n_atoms)
            if len(self.lines[1]) == 0:
                self.lines[1] = str("[MOLNAME]")
            self.lines += [""] * n_atoms
            
            
            for i, atom in enumerate(atoms):
                line = "  " + str(atom.element)            
                line += "       " + "{: .{}f}".format(atom.coord[0], 5)
                line += "       " + "{: .{}f}".format(atom.coord[1], 5)
                line += "       " + "{: .{}f}".format(atom.coord[2], 5)
                line += " "
                self.lines[i+2] = line
                
        elif isinstance(atoms, AtomArrayStack):
            n_lines_per_model = atoms[0].shape[0]+2
            
            
            self.lines += [""] * n_lines_per_model*atoms.shape[0]
            
            for i, atoms_i in enumerate(atoms):
            
                self.lines[i*n_lines_per_model] = str(atoms[0].shape[0])
                self.lines[i*n_lines_per_model+1] = " " + str(i)
                    
                for j, atom in enumerate(atoms_i):
                    line = "  " + str(atom.element)            
                    line += "       " + "{: .{}f}".format(atom.coord[0], 5)
                    line += "       " + "{: .{}f}".format(atom.coord[1], 5)
                    line += "       " + "{: .{}f}".format(atom.coord[2], 5)
                    line += " "
                    self.lines[i*n_lines_per_model+j+2] = line                
                


