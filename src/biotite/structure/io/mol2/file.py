# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mol2"
__author__ = "Benjamin E. Mayer"
__all__ = ["MOL2File","supported_charge_types", "supported_mol_types"]

import numpy as np
import biotite.structure as struc
from biotite.structure.error import BadStructureError
from ...atoms import AtomArray,AtomArrayStack, Atom, BondList
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
biotite_bonds_to_sybyl = {
    1: "1",
    2: "2",           
    3: "3",
    5: "ar",
    0: "un",                    
}

supported_charge_types = [
    "NO_CHARGES", "DEL_RE", "GASTEIGER", 
    "GAST_HUCK", "HUCKEL", "PULLMAN", 
    "GAUSS80_CHARGES", "AMPAC_CHARGES", 
    "MULLIKEN_CHARGES", "DICT_CHARGES", 
    "MMFF94_CHARGES", "USER_CHARGES"
]
    
    
supported_mol_types = [
    "SMALL", "BIOPOLYMER", "PROTEIN", "NUCLEIC_ACID", "SACCHARIDE"
]    
    
sybyl_status_bit_types = [
    "system", "invalid_charges", "analyzed", "substituted", "altered", "ref_angle"
]
    
def get_sybyl_atom_type(atom, bonds, atom_id):
    """
    This function is meant to translate all occuring atoms into sybyl atom
    types based on their element and bonds. This does however not work yet,
    therefore which is why currently the sybyl column is simply filled with
    a static content in order to not mess up rereading from a file written
    from the MOL2File class.
    
    Parameters
    ----------
        atom : Atom
            The Atom object which is to be translated 
            into sybyl atom_type notation
        
        bonds: BondList
            The BondList object of the respective bonds in the AtomArray.
            Necessary as the sybyl atom types depend on the hybridiziation 
            heuristic.
            
        atom_id: int
            The id of the current atom that is used in the bond list.                    
            
    Returns
    -------     
        sybyl_atom_type : str
            The name of the atom based on hybridization and element according
            to the sybyl atom types.       
        
    """
    atom_bonds, types = bonds.get_bonds(atom_id)    
    
    if atom.element == "C":                          
        if 5 in types:
            return "C.ar"
        else:
            if len(atom_bonds) == 3:
                return "C.1"
            elif len(atom_bonds) == 2:
                return "C.2"
            elif len(atom_bonds) == 1:
                return "C.3"                         
#            else:
#                msg = "No supported sybyl Atom type for Atom " + str(atom)
#                raise ValueError(msg)                  
            return "C.3"
    if atom.element == "N":                                   
        if 5 in types:
            return "N.ar"
        else:       
            if len(atom_bonds) == 3:
                return "N.1"
            elif len(atom_bonds) == 2:
                return "N.2"
            elif len(atom_bonds) == 1:
                return "N.3"               
    if atom.element == "O":
        if len(atom_bonds) == 2:
            return "O.3"
        elif len(atom_bonds) == 1:
            return "O.2"   
    
    if atom.element == "S":
        return "S.3"
    if atom.element == "P":
        return "P.3"
    if atom.element == "F":
        return "F"
    if atom.element == "H":
        return "H"
    if atom.element == "Li":
        return "Li"                                                        
    if atom.element == "Na":
        return "Na"                 
    if atom.element == "Mg":
        return "Mg"            
    if atom.element == "Al":
        return "Al"        
    if atom.element == "Si":
        return "Si"        
    if atom.element == "K":
        return "K"        
    if atom.element == "Ca":
        return "Ca"        
    if atom.element == "Cr":
        return "Cr.th"
    if atom.element == "Mn":
        return "Mn"
    if atom.element == "Fe":
        return "Fe"
    if atom.element == "Co":
        return "Co.oh"
    if atom.element == "Cu":
        return "Cu"     
    if atom.element == "Cl":
        return "Cl"
    if atom.element == "Br":
        return "Br"
    if atom.element == "I":
        return "I"
    if atom.element == "Zn":
        return "Zn"
    if atom.element == "Se":
        return "Se"
    if atom.element == "Mo":
        return "Mo"
    if atom.element == "Sn":
        return "Sn"
        
        
        
        
        
                                               

                                                            
        
            


class MOL2File(TextFile):
    """
    This class represents a file in MOL2 format. 
    
    Notes: 
        - For multiple models the same header for all models is assumed.
        - As biotites charge field in the AtomArray and AtomArrayStack class
          is typed as int this class adds a field charges containing the 
          real valued charges contained in MOL2 files.
        - The heuristic function for deriving sybyl atom types doesn't work
          yet, for now we only write C.ar in the according column as the
          sybyl atom type is one of the necessary fields for a complete 
          MOL2File.                   
    
    References
    ----------
    
    .. footbibliography::
    
    
    
    Examples
    --------

    >>> from os.path import join
    >>> mol_file = MOL2File.read(join(path_to_structures, "molecules", "nu7026.mol2"))
    >>> atom_array = mol2_file.get_structure()
    >>> print(atom_array)   
            1  UNL        O         0.867    0.134    0.211
            1  UNL        C         2.090    0.096    0.147
            1  UNL        C         2.835   -1.164    0.071
            1  UNL        C         2.156   -2.382    0.096
            1  UNL        C         2.879   -3.572    0.063
            1  UNL        C         4.282   -3.558    0.021
            1  UNL        C         4.974   -2.321   -0.013
            1  UNL        C         6.385   -2.347   -0.040
            1  UNL        C         7.089   -3.553   -0.030
            1  UNL        C         6.399   -4.759   -0.002
            1  UNL        C         5.004   -4.762    0.020
            1  UNL        C         4.225   -1.121   -0.006
            1  UNL        O         4.909    0.073   -0.087
            1  UNL        C         4.218    1.277    0.018
            1  UNL        C         2.884    1.328    0.147
            1  UNL        N         5.001    2.423   -0.069
            1  UNL        C         4.820    3.609    0.790
            1  UNL        C         4.974    4.886   -0.048
            1  UNL        O         6.238    4.918   -0.733
            1  UNL        C         6.369    3.772   -1.593
            1  UNL        C         6.282    2.457   -0.799
     



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
        self.charges = None
        
        self.ind_molecule = -1
        self.ind_atoms = -1
        self.ind_bonds = -1
        self.sybyl_atom_types = None
        

    def check_valid_mol2(self):
    
        #self.ind_molecule = np.where("@<TRIPOS>MOLECULE" in self.lines)[0]       
        self.ind_molecule = [
                i for i, x in enumerate(self.lines) if "@<TRIPOS>MOLECULE" in x
        ]
        if len(self.ind_molecule) == 0:
            raise ValueError(
                "Mol2 File doesn't contain a MOLECULE section, therefore"
                "it is not possibe to parse this file"
            )
            
        
        self.ind_atoms = [
                i for i, x in enumerate(self.lines) if "@<TRIPOS>ATOM" in x
        ]
        if len(self.ind_atoms) == 0:
            raise ValueError(
                "Mol2 File doesn't contain a ATOM section, therefore"
                "it is not possibe to parse this file"
            )
 
            
        self.ind_bonds = [
                i for i, x in enumerate(self.lines) if "@<TRIPOS>BOND" in x
        ]
        if len(self.ind_bonds) == 0:
            raise ValueError(
                "Mol2 File doesn't contain a BOND section, therefore"
                "it is not possibe to parse this file"
            )
            
        if len(self.ind_molecule) != len(self.ind_atoms):
            raise ValueError(
                "Mol2 File doesn't contain as many MOLECULE sections as it does"
                "contain ATOM sections"
                
            )   
        if len(self.ind_molecule) != len(self.ind_bonds):
            raise ValueError(
                "Mol2 File doesn't contain as many MOLECULE sections as it does"
                "contain BOND sections"
                
            ) 
        if len(self.ind_bonds) != len(self.ind_atoms):
            raise ValueError(
                "Mol2 File doesn't contain as many BOND sections as it does"
                "contain ATOM sections"
                
            )                                              
                                      
       
    
    def get_header(self):
        """
        Get the header from the MOL2 file, if the file contains multiple
        models the assumption is made that those all have the same header
        as a AtomArrayStack of different molecules can't be buil anyways.
        
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
                    
        self.check_valid_mol2()                                                    
        self.mol_name = self.lines[self.ind_molecule[0]+1]
                
        numbers_line = self.lines[self.ind_molecule[0]+2]
        numbers_parsed = [int(x) for x in numbers_line.strip().split(" ")]

        self.num_atoms = numbers_parsed[0]
        if len(numbers_parsed) > 1:
            self.num_bonds = numbers_parsed[1]
        if len(numbers_parsed) > 2:            
            self.num_subst = numbers_parsed[2]
        if len(numbers_parsed) > 3:            
            self.num_feat  = numbers_parsed[3]
        if len(numbers_parsed) > 4:            
            self.num_sets  = numbers_parsed[4]
        

        self.mol_type    = self.lines[self.ind_molecule[0]+3]
        self.charge_type = self.lines[self.ind_molecule[0]+4]
        self.status_bits = self.lines[self.ind_molecule[0]+5]
        
        if "@" not in self.lines[self.ind_molecule[0]+6]:
            self.mol_comment = self.lines[self.ind_molecule[0]+6]
        
        return (
            self.mol_name, self.num_atoms, self.mol_type,
            self.charge_type, self.num_bonds, self.num_subst,   
            self.num_feat, self.num_sets, self.status_bits,
            self.mol_comment
        )
                        
           


    def set_header(self, mol_name, num_atoms, mol_type, charge_type,
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
            If charge_type is NO_CHARGES the according charge column 
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
        
        if num_bonds >= 0:
            self.num_bonds      = num_bonds
        if num_subst >= 0:            
            self.num_subst      = num_subst
            
        if num_feat >= 0:            
            self.num_feat       = num_feat
    
        if num_sets >=0:            
            self.num_sets       = num_sets
        
        header = [
            mol_name, num_atoms, mol_type, charge_type,
            num_bonds, num_subst, num_feat, num_sets,
            status_bits, mol_comment
        ]
        print(header)
        
        if mol_type != "":
            cond = mol_type in supported_mol_types
            if not cond:
                msg  = "The specified molecule type ["+str(charge_type) +"] \n"
                msg += " is not among the supported molecule types: \n"
                msg += "" + str(supported_mol_types) + "\n"  
                msg += "header :: " + str(header) + " \n"       
                raise ValueError(msg)

        self.mol_type       = mol_type
        
        if charge_type != "":            
            cond = charge_type in supported_charge_types
            if not cond:
                msg  = "The specified charge type ["+str(charge_type) +"] "
                msg += " is not among the supported charge types: \n"
                msg += str(supported_charge_types) + "\n"
                raise ValueError(msg)                          
                
        self.charge_type    = charge_type
        
        self.status_bits    = status_bits
        self.mol_comment    = mol_comment
        
        #if len(self.lines)==0:
        if self.status_bits != "":
            self.lines = [""]*5
            self.lines[4] = self.status_bits
        else:
            self.lines = [""]*6  
            self.lines[4] = self.status_bits
            self.lines[5] = self.mol_comment
            
        self.lines[0] = "@<TRIPOS>MOLECULE"
        self.ind_molecule = [0]
        self.lines[1] = self.mol_name

        line = " " + str(self.num_atoms)
        if self.num_bonds >= 0:
            line += " " + str(self.num_bonds)
            
            if self.num_subst >= 0:
                line += " " + str(self.num_subst)
        
                if self.num_feat >= 0:
                    line += " " + str(self.num_feat)
                    
                    if self.num_sets >= 0:
                        line += " " + str(self.num_sets)
    
        self.lines[2] = line      
        self.lines[3] = self.mol_type
        self.lines[4] = self.charge_type
        
        self.lines[5] = self.status_bits                                  
        if self.status_bits != "":
            self.lines[6] = self.mol_comment
                                                     
        self.ind_atoms = [len(self.lines)]                                                       
        

    def get_structure(self):
        """
        Get an :class:`AtomArray` from the MOL2 file.
        
        Returns
        -------
        array : AtomArray or AtomArrayStack
            This :class:`AtomArray` contains the optional ``charge``
            annotation and has an associated :class:`BondList`.
            All other annotation categories, except ``element`` are
            empty.
        """
            
        self.get_header()        
        atom_array_stack = []
        # instantiate atom array and bonds based on number of atoms information
        
        self.sybyl_atom_types = []
        for i in range(len(self.ind_atoms)):
        
            atoms = AtomArray(self.num_atoms)    
            
            if self.charge_type != "NO_CHARGES":
                atoms.add_annotation("charge", int)
                if self.charges is None:
                    self.charges = np.zeros(
                        self.num_atoms
                    ).reshape(
                        (1, self.num_atoms)
                    )
                    
                else:
                    self.charges = np.vstack(
                        (
                            self.charges, 
                            np.zeros(self.num_atoms)
                        )                            
                    )                    
                
            bonds = BondList(self.num_atoms)
            
            # 
            #   Iterate through all the atom lines by stating from line after
            #   @<TRIPOS>ATOM until line starting with @ is reached.
            #   All lines in between are assumed to be atom lines and are parsed
            #   into atoms accodringly.
            #
            index = self.ind_atoms[i]+1
            j = 0
            atom_type_sybl_row = [""]*self.num_atoms
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
#                    print(line)
#                    print(charge)                       
                if len(line) > 9:
                    status_bits     = line[9]
                    
                    
                  
                if self.charge_type != "NO_CHARGES": 
                    self.charges[i][j] = charge                                
                    atom_i = Atom(
                        [x_coord, y_coord, z_coord],
                        charge=int(np.rint(charge))
                    )     
                else:
                    atom_i = Atom(
                        [x_coord, y_coord, z_coord],                               
                    )                                    
                
                atom_i.atom_id  = atom_id                
                atom_i.element  = atom_name                        
                atom_i.res_id   = subst_id
                atom_i.res_name = subst_name
                
                atoms[j] = atom_i  
                atom_type_sybl_row[j] = atom_type_sybl 
                index += 1   
                j += 1
                

            # 
            #   Iterate through all the bond lines by stating from line after
            #   @<TRIPOS>BOND until line starting with @ is reached.
            #   All lines in between are assumed to be atom lines and are parsed
            #   into atoms accodringly.
            #
            index = self.ind_bonds[i] +1
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
                    bonds.add_bond(
                        origin_atom_id-1,
                        target_atom_id-1,
                        bond_typ
                    )                
                    
                index += 1                                  
                                                                     
            atoms.bonds = bonds
            atom_array_stack.append(atoms)
            self.sybyl_atom_types.append(atom_type_sybl_row)
            
        if len(atom_array_stack) == 1:
            return atom_array_stack[0]
        else:
            return struc.stack(atom_array_stack)                        
             
            
    def append_atom_array(self, atoms, charges=None):
            
        n_atoms = atoms.shape[0]                 
        
        self.lines.append("@<TRIPOS>ATOM")    
        atoms_has_atom_ids = hasattr(atoms, "atom_id")
        
        if charges is not None:
            assert len(charges) == len(atoms)
            
        for i, atom in enumerate(atoms):
            
            atom_id = i+1
            if atoms_has_atom_ids:
                atom_id = atom.atom_id
                
            line  = "{:>7}".format(atom_id)
            line += "  " + atom.element
            line += "{:>16.4f}".format(atom.coord[0])
            line += "{:>10.4f}".format(atom.coord[1])
            line += "{:>10.4f}".format(atom.coord[2])                                          
            line += " {:<8}".format(
                get_sybyl_atom_type(
                    atom, atoms.bonds, i
                )
#                "C.ar"
            )
            if atom.res_id != 0:                                          
                line += str(atom.res_id)
                
                if atom.res_name != "":
                    line += "  " + str(atom.res_name)
                    
                    if self.charge_type != "NO_CHARGES":
                        line += "       "        
#                        print("charged")            
                        if charges is not None:  
#                            print(
#                                " using user defined charges " +str(charges[i])
#                            )                                                                                  
                            line += " {: .{}f}".format(charges[i], 4)
                        else:
                            line += " {: .{}f}".format(atom.charge, 4)                        
                                                        
                        
            self.lines.append(line)                                                                                                                                
        
            
        self.lines.append("@<TRIPOS>BOND")                
        for i, bond in enumerate(atoms.bonds.as_array()):
            line  = "{:>6}".format(i+1)
            line += "{:>6}".format(bond[0]+1)
            line += "{:>6}".format(bond[1]+1)
            line += "{:>5}".format(biotite_bonds_to_sybyl[bond[2]])
            self.lines.append(line)

    def set_structure(self, atoms):
        """
        Set the :class:`AtomArray` for the file.
        
        Parameters
        ----------
        array : AtomArray or 
            The array to be saved into this file.
            Must have an associated :class:`BondList`.
        """     
        
        if len(self.lines) < 5:
            msg  = "Header not valid, less then the minimum amount of lines in"
            msg += "header :: " +str(self.lines)            
            raise ValueError(msg)
                        
        header_lines = self.lines[:self.ind_atoms[0]]

        # since setting new structure delete all previously stored
        # structure information if any
        self.lines = self.lines[:self.ind_atoms[0]]        
        
        if isinstance(atoms, AtomArray):

            
            if atoms.bonds is None:
                raise BadStructureError(
                    "Input AtomArrayStack has no associated BondList"
                )
                
            if self.charge_type != "NO_CHARGES" and atoms.charge is None:

                msg  = "Specified charge type " + str(self.charge_type) + ".\n"
                msg += "but given AtomArray has no charges"
                raise ValueError(msg)     
                
            if self.num_atoms != atoms.shape[0]:
                msg  = "Mismatch between number of atoms in header ["
                msg += str(self.num_atoms) + "] and number of atoms in given"
                msg += "AtomArray [" + str(atoms.shape[0]) + "]"
                raise ValueError(msg)                                   
        
#            print(" charges ::")
#            print(self.charges)
#            print("")
            if self.charges is not None:                                            
                self.append_atom_array(atoms, self.charges)                
            else:
                self.append_atom_array(atoms)                
                            
        
        elif isinstance(atoms, AtomArrayStack):      
        
            if atoms.bonds is None:
                raise BadStructureError(
                    "Input AtomArrayStack has no associated BondList"
                )
            if self.charge_type != "NO_CHARGES" and atoms.charge is None:

                msg  = "Specified charge type " + str(self.charge_type) + ".\n"
                msg += "but one of given AtomArrays has no charges"
                raise ValueError(msg)                   
                
            if self.num_atoms != atoms.shape[1]:
                msg  = "Mismatch between number of atoms in header ["
                msg += str(self.num_atoms) + "] and number of atoms in given"
                msg += "AtomArrayStack [" + str(atoms.shape[1]) + "]"
                raise ValueError(msg)                
                    
            n_models    = atoms.shape[0]
            n_atoms     = atoms[0].shape[0]   
            n_bonds     = atoms[0].bonds.as_array().shape[0]         

            
            for i, atoms_i in enumerate(atoms):                          
                                  
            
                if i > 0:          
                    for l in header_lines:
                        self.lines.append(l)

#                print(" charges ::")
#                print(self.charges)
#                print("")
                if self.charges is not None:
                    self.append_atom_array(atoms_i, self.charges[i])                
                else:
                    self.append_atom_array(atoms_i)               

    def set_charges(self, charges):            
        """
        Set a partial charges array that will be used for writing the mol2 file.
                              
        """
        
#        if not self.charges is None and self.charges.shape != charges.shape:
#            msg  = "Can only assign charges of same shape as already within"
#            msg += "Mol2 file 
#            raise ValueError(msg)
#            
#        
#        if self.num_atoms != -1:

#        print("setting self.charges :: " + str(self.charges))
#        print("to                   :: " + str(charges))

        self.charges = charges
            

    def get_charges(self):
        
        if self.charge_type == "NO_CHARGES":
            raise ValueError(
                "Can not get charges from mol2 file where NO_CHARGES set."
            )
        
        if self.charges is None:
            _ = self.get_structure()
            
            
        if len(self.charges) == 1:
            return self.charges[0]
        else:                               
            return self.charges            
            
    
     
            


