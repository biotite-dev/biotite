# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.sdf"
__author__ = "Benjamin E. Mayer"
__all__ = ["SDFFile"]

import numpy as np
from ...atoms import AtomArray, Atom
from ....file import TextFile


# Number of header lines
N_HEADER = 2


class SDFFile(TextFile):
    """
    This class represents a file in SDF format, 
    
    References
    ----------
    
    .. footbibliography::
    
    Examples
    --------

    >>> from os.path import join
    >>> mol_file = SDFFile.read(join(path_to_structures, "molecules", "BENZ.sdf"))
    >>> atom_array = sdf_file.get_structure()
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
    
    def get_header(self):
        """
        Get the header from the SDF file.
        
        Returns
        -------
        mol_name : str
            The name of the molecule.
        initials : str
            The author's initials.
        program : str
            The program name.
        time : datetime
            The time of file creation.
        dimensions : str
            Dimensional codes.
        scaling_factors : str
            Scaling factors.
        energy : str
            Energy from modeling program.
        registry_number : str
            MDL registry number.
        comments : str
            Additional comments.
        """
        atom_number     = int(self.lines[0].strip())
        mol_name        = self.lines[1].strip()
        return atom_number, mol_name


    def set_header(self, mol_name):
        """
        Set the header for the SDF file.
        
        Parameters
        ----------
        mol_name : str
            The name of the molecule.
  
        """
        
        if(len(self.lines) > 2):
            
            self.lines[1] = str(mol_name) + "\n"
            self.lines[0] = str(len(self.lines)-2)+ "\n"
        
        else:
            raise ValueError(
                    "Can not set header of an empty SDFFile"
                    "Use set_structure first, so that number of atoms"
                    "can be derived from set structure"
                )


    def get_structure(self):
        """
        Get an :class:`AtomArray` from the SDF file.
        
        Returns
        -------
        array : AtomArray
            This :class:`AtomArray` contains the optional ``charge``
            annotation and has an associated :class:`BondList`.
            All other annotation categories, except ``element`` are
            empty.
        """
        
        if len(self.lines) > 2:
        
        
            array = AtomArray(len(self.lines)-2)
            
            for i, line in enumerate(self.lines[2:]):
                line_parsed = [x for x in line.strip().split(" ") if x!= '']
                
                atom = Atom(
                    [
                        float(line_parsed[1]),
                        float(line_parsed[2]),
                        float(line_parsed[3]),                                                                        
                    ]
                )
                atom.element = line_parsed[0]                
                array[i] = atom
                
            return array                
                
                
            
        else:
        
            raise ValueError(
                "Trying to get_structure from empty SDFFile"
            )            
        
        

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
        
        
        for i, atom in enumerate(atoms):
            line = "  " + atom.element 
            line += "       " + "{: .{}f}".format(atom.coord[0], 5)
            line += "       " + "{: .{}f}".format(atom.coord[1], 5)
            line += "       " + "{: .{}f}".format(atom.coord[2], 5)
            line += " "
            self.lines[i+2] = line


