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
    This class represents a file in XYZ format, 
    
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
        self.mol_names = None
        self.atom_numbers = None
        self.model_start_inds = None
        self.structures = None
        
    def update_start_lines(self):

        # Line indices where a new model starts -> where number of atoms
        # is written down as number
    
        self.model_start_inds = np.array(
            [i for i in range(len(self.lines))
             if self.lines[i].strip().isdigit()],
             dtype=int
         )
#         
#        print(" ... pre processing ... ")                            
#        print("|model_start_inds| :: " + str(self.model_start_inds))
                     
        if self.model_start_inds.shape[0] > 1:             
            # if the mol name line only contains an integer
            # these lines will appear in model_start_inds 
            # if calculated as above, therfore we purge all 
            # indices that have a distance of 1 to their previous index
            # (this will not work with a file containing multiple models
            #  with solely one coordinate / atom per model )                                                     
            self.model_start_inds = self.model_start_inds[
                np.concatenate(
                    (
                        np.array([0],dtype=int), 
                        np.where(
                            self.model_start_inds[:-1]
                                -
                            self.model_start_inds[1:] 
                                != 
                            -1
                        )[0]+1
                    )
                )
            ] 
        elif self.model_start_inds.shape[0] == 2:                            
            self.model_start_inds = self.model_start_inds[:1]            
    
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
        self.update_start_lines()
                        
        # parse all atom_numbers into integers
        if self.atom_numbers is None:
            self.atom_numbers = [
                int(
                    self.lines[i].strip().strip(" ")
                ) for i in self.model_start_inds
            ]
          
        # parse all lines containing names             
        if self.mol_names is None:
            self.mol_names = [
                self.lines[i+1].strip() for i in self.model_start_inds
            ]        
        
        
        return self.atom_numbers, self.mol_names


    def set_header(self, mol_name):
        """
        Set the header for the XYZ file.
        
        Parameters
        ----------
        mol_name : str
            The name of the molecule.
  
        """
        
        if(len(self.lines) > 2):
            
            self.lines[1] = str(mol_name)
            self.lines[0] = str(len(self.lines)-2)
            
            self.mol_names = [mol_name]
            self.atom_numbers = [int(self.lines[0])]
            
            self.update_start_lines()
        
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
#        print("atom_numbers :: " + str(atom_number) + " :: " +str(len(atom_number)))
#        print("names        :: " + str(names)+ " :: " +str(len(names)))
#        
        # set a default head if non present since
        # the number of lines will be calculated from the number
        # of atoms field in the file (counts how many lines with numbers
        # there are within the file which are not name lines)
        if len(names) == 0 or len(atom_number) == 0:
#            print("self.set_header('[MOLNAME]')")
            self.set_header("[MOLNAME]")  
#            print(self.lines)
           
        self.update_start_lines()
                
                            
        # parse all atom_numbers into integers
        if self.atom_numbers is None:
            self.atom_numbers = [
                int(
                    self.lines[i].strip().strip(" ")
                ) for i in self.model_start_inds
            ]
          
        # parse all lines containing names             
        if self.mol_names is None:
            self.mol_names = [
                self.lines[i+1].strip() for i in self.model_start_inds
            ]
            

        # parse all coordinates        
        if self.structures is None:                                   
            
          
            
            array_stack = []
            
#            print("|model_start_inds| :: " + str(self.model_start_inds))
#            print(self.atom_numbers)
#            print("")
        
            for i, ind in enumerate(self.model_start_inds):
                ind_end = ind+2 + self.atom_numbers[i]
#                print("self.lines ::")
#                print(self.lines)      
#                print("")
#                print("ind     :: " +str(ind))            
#                print("ind_end :: " +str(ind_end))                    
                lines_cut = self.lines[ind:ind_end]  
#                print("lines_cut ::")  
#                print("")                                      
#                print(lines_cut)                 
                array = AtomArray(self.atom_numbers[i])
#                print("empty array ::")
#                print(array)
                
                if self.atom_numbers[i]+2 != len(lines_cut):
                    raise ValueError(
                        "Number of Atoms not matching with coordinate lines"
                        + ""
                        + " atom_number :: " + str(self.atom_numbers[i])
                        + ""
                        + ""
                        + " |lines_cut| :: " + str(len(lines_cut))
                        + " lines_cut   :: " + str(lines_cut)
                    )                         
                
                for j, line in enumerate(lines_cut[2:]):
#                        print(line)
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
                                                            
                    

#                print("filled array ::")
#                print(array)    
                
                array_stack.append(array)
            self.structures = struc.stack(array_stack)
                                        
                
   
                
                    
        
        if model is None:            
            return self.structures
        else:
            return self.structures[model]                          
        
        

    def set_structure(self, atoms):
        """
        Set the :class:`AtomArray` for the file.
        
        Parameters
        ----------
        array : AtomArray
            The array to be saved into this file.
            Must have an associated :class:`BondList`.
        """
        
        n_atoms = atoms.shape[0]
        self.lines += [""] * n_atoms
        
#        print(" trying atoms :: ")
#        print(atoms)
#        print()
        
        
        for i, atom in enumerate(atoms):
            line = "  " + str(atom.element)            
            #print(atom.coord)
            line += "       " + "{: .{}f}".format(atom.coord[0], 5)
            line += "       " + "{: .{}f}".format(atom.coord[1], 5)
            line += "       " + "{: .{}f}".format(atom.coord[2], 5)
            line += " "
            self.lines[i+2] = line


