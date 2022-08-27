# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mol2"
__author__ = "Benjamin E. Mayer"
__all__ = ["MOL2File"]

import numpy as np
import biotite.structure as struc
from ...atoms import AtomArray, Atom, BondList
from ....file import TextFile





sybyl_to_biotite_bonds = {

    "1": struc.BondType.SINGLE,
    "2": struc.BondType.DOUBLE,
    "3": struc.BondType.TRIPLE,
    "am": None,             # amide not really supported in biotite yet
    "ar": struc.BondType.AROMATIC_SINGLE,  # questionable if this is okay since we have up to 3 formats for aromatic bonds
    "du": None,             # no idea what dummy is
    "un": struc.BondType.ANY,
    "nc": None,
}
    
    
    


class MOL2File(TextFile):
    """
    This class represents a file in MOL2 format, 
    
    References
    ----------
    
    .. footbibliography::
    
    Examples
    --------

    >>> from os.path import join
    >>> mol_file = MOL2File.read(join(path_to_structures, "molecules", "nu7026.mol2"))
    >>> atom_array = mol2_file.get_structure()
    >>> print(atom_array)    



    """
    
    def __init__(self):
        super().__init__()
        self.mol_name = ""
        self.num_atoms = -1
        self.num_bonds = -1
        self.num_subst = -1
        self.num_feat = -1
        self.num_sets = -1
        self.mol_type = ""
        self.charge_type = ""
        self.status_bits = ""
        self.mol_comment = ""        
        
        self.ind_molecule = -1
        self.ind_atoms = -1
        self.ind_bonds = -1
        

    def check_valid_mol2(self):
    
        #self.ind_molecule = np.where("@<TRIPOS>MOLECULE" in self.lines)[0]       
        self.ind_molecule = [
                i for i, x in enumerate(self.lines) if "@<TRIPOS>MOLECULE" in x
        ]
        if len(self.ind_molecule) != 1:
            raise ValueError(
                "Mol2 File doesn't contain a MOLECULE section, therefore"
                "it is not possibe to parse this file"
            )
        else:
            self.ind_molecule = self.ind_molecule[0] 
            
        
        self.ind_atoms = [
                i for i, x in enumerate(self.lines) if "@<TRIPOS>ATOM" in x
        ]
        if len(self.ind_atoms) != 1:
            raise ValueError(
                "Mol2 File doesn't contain a ATOM section, therefore"
                "it is not possibe to parse this file"
            )
        else:
            self.ind_atoms = self.ind_atoms[0]    
            
        self.ind_bonds = [
                i for i, x in enumerate(self.lines) if "@<TRIPOS>BOND" in x
        ]
        if len(self.ind_bonds) != 1:
            raise ValueError(
                "Mol2 File doesn't contain a BOND section, therefore"
                "it is not possibe to parse this file"
            )
        else:
            self.ind_bonds = self.ind_bonds[0]    
            
#        print(self.ind_molecule)
#        print(self.ind_atoms)
#        print(self.ind_bonds)                                      
       
    
    def get_header(self):
        """
        Get the header from the MOL2 file.
        
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
        
#        if self.record_inds is None:

#            self.record_inds = [
#                i for i, x in enumerate(self.lines) if "@" in x
#            ]
#            #print(self.record_inds)
#            
                    
        self.check_valid_mol2()
                                    
        
        #print(ind_molecule)
        self.mol_name = self.lines[self.ind_molecule+1]
        
        numbers_line = self.lines[self.ind_molecule+2]
        numbers_parsed = [int(x) for x in numbers_line.strip().split(" ")]

        self.num_atoms = numbers_parsed[0]
        self.num_bonds = numbers_parsed[1]
        self.num_subst = numbers_parsed[2]
        self.num_feat  = numbers_parsed[3]
        self.num_sets  = numbers_parsed[4]
        

        self.mol_type    = self.lines[self.ind_molecule+3]
        self.charge_type = self.lines[self.ind_molecule+4]
        self.status_bits = self.lines[self.ind_molecule+5]
        
        if "@" not in self.lines[self.ind_molecule+6]:
            self.mol_comment = self.lines[self.ind_molecule+6]
        
        return (
            self.mol_name, 
            self.num_atoms, self.num_bonds, self.num_subst,   
            self.num_feat, self.num_sets, 
            self.mol_type, self.charge_type, self.status_bits,
            self.mol_comment
        )
                        
           


    def set_header(self, mol_name, num_atoms, mol_type, charge_type
            num_bonds=-1, num_subst=-1, num_feat=-1, num_sets=-1, 
            status_bits="", mol_comment=""):
        """
        Set the header for the MOL2 file according the following structure:
        
        mol_name
        num_atoms [num_bonds [num_subst [num_feat[num_sets]]]]
        mol_type
        charge_type
        [status_bits
        [mol_comment]]
        
        taken from https://chemicbook.com/2021/02/20/mol2-file-format-explained-for-beginners-part-2.html
        
        Parameters
        ----------
        mol_name : str
            The name of the molecule.
        num_atoms: int
            The number of atoms in the molecule.
        mol_type: str
            The molecule type given as a string may be either SMALL,
            BIOPOLYMER, PROTEIN, NUCLEIC_ACID or SACCHARIDE
        charge_type: str
            Specifies the used method for calculating the charges if any
            are given. May be either NO_CHARGES, DEL_RE, GASTEIGER, 
            GAST_HUCK, HUCKEL, PULLMAN, GAUSS80_CHARGES, AMPAC_CHARGES, 
            MULLIKEN_CHARGES, DICT_CHARGES, MMFF94_CHARGES or USER_CHARGES.
            If charget_type is NO_CHARGES the according charge column 
            will be ignored even if charges are given.
        num_bonds: int, optional
            Number of bonds given as integer, if any.
        num_subst: int, optional
            
        num_feat: int, optional
        
        num_sets: int, optional
        
        status_bits: str, optional
        
        mol_comment: str, optional                            
                                                                                                                                          
  
        """
        
        self.mol_name       = mol_name
        self.num_atoms      = num_atoms
        self.num_bonds      = num_bonds
        self.num_subst      = num_subst
        self.num_feat       = num_feat
        self.num_sets       = num_sets
        self.mol_type       = mol_type
        self.charge_type    = charge_type
        self.status_bits    = status_bits
        self.mol_comment    = mol_comment
        
        if len(self.lines)==0:
            if self.mol_comment == "":
                self.lines = [""]*5
                self.lines[4] = self.status_bits
            else:
                self.lines = [""]*6  
                self.lines[4] = self.status_bits
                self.lines[5] = self.mol_comment                                             
        
        
        
        if(len(self.lines) > 2):
            
            self.lines[1] = str(mol_name) + "\n"
            self.lines[0] = str(len(self.lines)-2)+ "\n"
        
        else:
            raise ValueError(
                    "Can not set header of an empty MOL2File"
                    "Use set_structure first, so that number of atoms"
                    "can be derived from set structure"
                )


    def get_structure(self):
        """
        Get an :class:`AtomArray` from the MOL2 file.
        
        Returns
        -------
        array : AtomArray
            This :class:`AtomArray` contains the optional ``charge``
            annotation and has an associated :class:`BondList`.
            All other annotation categories, except ``element`` are
            empty.
        """
        
#        self.check_valid_mol2()        
        self.get_header()        
        
        # instantiate atom array and bonds based on number of atoms information
        atoms = AtomArray(self.num_atoms)
        bonds = BondList(self.num_atoms)
        
        # 
        #   Iterate through all the atom lines by stating from line after
        #   @<TRIPOS>ATOM until line starting with @ is reached.
        #   All lines in between are assumed to be atom lines and are parsed
        #   into atoms accodringly.
        #
        index = self.ind_atoms+1
        i = 0
#        print(" start index :: " +str(index))
        while "@" not in self.lines[index]:
        
        
            line = [x for x in self.lines[index].strip().split(" ") if x != ''] 
             
            
            atom_id         = int(line[0])
            atom_name       = line[1]
            x_coord         = float(line[2])
            y_coord         = float(line[3])
            z_coord         = float(line[4])                        
            atom_type_sybl  = line[5]
            subst_id        = -1
            subst_name      = ""
            charge          = 0.0
            status_bits     = ""
            
            if len(line) > 6:
                subst_id        = int(line[6])                
            if len(line) > 7:                
                subst_name      = line[7]
            if len(line) > 8:                
                charge          = float(line[8])
            if len(line) > 9:
                status_bits     = line[9]
                
            atom_i = Atom(
                [x_coord, y_coord, z_coord],
                
            )     
            atom_i.atom_id  = atom_id
            if self.charge_type != "NO_CHARGES":
                atom_i.charge = charge
            atom_i.element  = atom_name                        
            atom_i.res_id   = subst_id
            atom_i.res_name = subst_name
            
            atoms[i] = atom_i                          
            index += 1   
            i += 1
            

        # 
        #   Iterate through all the bond lines by stating from line after
        #   @<TRIPOS>BOND until line starting with @ is reached.
        #   All lines in between are assumed to be atom lines and are parsed
        #   into atoms accodringly.
        #
        index = self.ind_bonds +1
        while index < len(self.lines) and "@" not in self.lines[index]:  
        
            line = [x for x in self.lines[index].strip().split(" ") if x != '']
            
            bond_id             = int(line[0])
            origin_atom_id      = int(line[1])
            target_atom_id      = int(line[2])
            bond_typ            = sybyl_to_biotite_bonds[str(line[3])]
            status_bits         = ""
            
            if len(line) > 4:
                status_bits     = str(line[4])
            
            if bond_typ is not None:
#                print(
#                    " adding bond ( " 
#                    + str(origin_atom_id-1) + ", " 
#                    + str(target_atom_id-1) + ", "
#                    + str(bond_typ) 
#                    + ")"
#                )                    
                    
                bonds.add_bond(
                    origin_atom_id-1,
                    target_atom_id-1,
                    bond_typ
                )                
                
            index += 1                                  
                                                

        atoms.bond_list = bonds
        return atoms            
             
            
        

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


