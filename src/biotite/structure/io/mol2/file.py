# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mol2"
__author__ = "Benjamin E. Mayer"
__all__ = ["MOL2File"]

import numpy as np
import biotite.structure as struc
from biotite.structure.error import BadStructureError
from ...atoms import AtomArray, AtomArrayStack, Atom, BondList
from ....file import TextFile


class MOL2File(TextFile):
    """
    This class represents a file in MOL2 format.
    If a completly new instance of this file is to be filled from an
    AtomArray or an AtomArrayStack, use set_structure first and modify
    the header afterwards with set_header. The reason for this is as
    follows: set_structure will assume a skeleton header and automatically
    generates as many lines as necessary for the structure as well as a
    header (ignoring the molecule_comment line) to be written.


    Notes:
        - For multiple models the same header for all models is assumed.
        - As biotites charge field in the AtomArray and AtomArrayStack class
          is typed as int this class adds a field charges containing the
          real valued charges contained in MOL2 files.
        - The heuristic function for deriving sybyl might not fully work,
          altough it does so for all the test cases.

    References
    ----------

    .. footbibliography::

    Examples
    --------
    >>> from os.path import join
    >>> mol_file = MOL2File.read(
    ...     join(path_to_structures, "molecules", "nu7026.mol2")
    ... )
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
    # mapping to get biotite bonds based on entry in MOL2File bonds
    sybyl_to_biotite_bonds = {
        "1": struc.BondType.SINGLE,
        "2": struc.BondType.DOUBLE,
        "3": struc.BondType.TRIPLE,
        "am": None,  # amide not really supported in biotite yet
        # questionable if this is okay since we have
        # up to 3 formats for aromatic bonds
        "ar": struc.BondType.AROMATIC_SINGLE,
        "du": None,  # no idea what dummy is
        "un": struc.BondType.ANY,
        "nc": None,
    }
    # revers of the mapping to write correct bonds
    biotite_bonds_to_sybyl = {
        1: "1",
        2: "2",
        3: "3",
        5: "ar",
        0: "un",
    }
    # list of supported values for the charge_type
    supported_charge_types = [
        "NO_CHARGES", "DEL_RE", "GASTEIGER",
        "GAST_HUCK", "HUCKEL", "PULLMAN",
        "GAUSS80_CHARGES", "AMPAC_CHARGES",
        "MULLIKEN_CHARGES", "DICT_CHARGES",
        "MMFF94_CHARGES", "USER_CHARGES"
    ]
    # list of supporte molecule types
    supported_mol_types = [
        "SMALL", "BIOPOLYMER", "PROTEIN", "NUCLEIC_ACID", "SACCHARIDE"
    ]
    # list of suppored sybyl bit types
    sybyl_status_bit_types = [
        "system", "invalid_charges",
        "analyzed", "substituted", "altered", "ref_angle"
    ]
    # field that contains all accepted two letter elements
    # this is used for deriving if sperate atom_name and element
    # columns should be generated. If only two letter atom_name entries in the
    # mol2 file are found that are contained here, the atom_name entry will be
    # assumed to only contain element names.
    elements_twoLetters = [
        "LI", "NA", "MG", "AL",
        "SI", "CA", "CR", "MN",
        "FE", "CO", "CU", "CL",
        "BR", "ZN", "SE", "MO",
        "SN", "AR"
    ]

    @staticmethod
    def transcribe_sybyl_atom_type(atom, atom_bonds, types, atom_id):
        """
        This function is meant to translate all occuring
        atoms into sybyl atom types based on their element and bonds. This
        mostly works now but some of the special cases are not implemented.
        Also it is not clear if there is some further special handling for
        the two letter element atoms or if their sybyl_atom_type is simply
        the same string with the second letter to lower case.

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
            The name of the atom based on hybridization and element
            according to the sybyl atom types.

        """
        N_bonds = np.where(atom_bonds != -1)[0].shape[0]
        if atom.element == "C":
            if 5 in types:
                return "C.ar"
            else:
                if N_bonds == 3:
                    return "C.1"
                elif N_bonds == 2:
                    return "C.2"
                elif N_bonds == 1:
                    return "C.3"
                return "C.3"

        if atom.element == "N":
            if 5 in types:
                return "N.ar"
            else:
                if N_bonds == 3:
                    return "N.1"
                elif N_bonds == 2:
                    return "N.2"
                elif N_bonds == 1:
                    return "N.3"

        if atom.element == "O":
            if N_bonds == 2:
                return "O.3"
            elif N_bonds == 1:
                return "O.2"
            else:
                msg = "Not implemented for element ["+str(atom.element) + "]"
                msg += " and bonds :: ["+str(atom_bonds)+"]"
                msg += " | types :: ["+str(types) + "]."
                msg += " | len(bonds) :: " + str(N_bonds)
                msg += " | 5 in types :: " + str(5 in types)
                raise ValueError(msg)

        if atom.element == "S":
            return "S.3"
        if atom.element == "P":
            return "P.3"
        if atom.element == "F":
            return "F"
        if atom.element == "H":
            return "H"
        if atom.element == "LI":
            return "Li"
        if atom.element == "NA":
            return "Na"
        if atom.element == "MG":
            return "Mg"
        if atom.element == "AL":
            return "Al"
        if atom.element == "SI":
            return "Si"
        if atom.element == "K":
            return "K"
        if atom.element == "CA":
            return "Ca"
        if atom.element == "CR":
            return "Cr.th"
        if atom.element == "MN":
            return "Mn"
        if atom.element == "FE":
            return "Fe"
        if atom.element == "CO":
            return "Co.oh"
        if atom.element == "CU":
            return "Cu"
        if atom.element == "CL":
            return "Cl"
        if atom.element == "BR":
            return "Br"
        if atom.element == "I":
            return "I"
        if atom.element == "ZN":
            return "Zn"
        if atom.element == "SE":
            return "Se"
        if atom.element == "MO":
            return "Mo"
        if atom.element == "SN":
            return "Sn"
        if atom.element == "AR":
            return "Ar"
        else:
            msg = "sybyl_atom_type not implemented for element ["
            msg += str(atom.element)
            msg += "] " + str(atom)
            raise ValueError(msg)

    @staticmethod
    def atom_name_to_element(atom_name, sybyl_atom_type):
        """
        This function gets a recorded atom name and sybyl_atom_type and returns
        you the according element that this pair should have.
        For example it is not possible to arrive from 'CA' if this should be
        calcium or the c-alpha Atom of a protein backbone. However together
        with the sybyl_atom_type it can be distinguished.
        """
        # a local function for generating error messages
        def get_error_msg(atom_name, sybyl_atom_type):
            msg = "Not implemented for given atom_name :: "
            msg += str(atom_name)+". "
            msg += "And given sybyl_atom_type          :: "
            msg += str(sybyl_atom_type) + "."
            return msg

        carbon_types = ["C.ar", "C.1", "C.2", "C.3"]
        if len(atom_name) == 1:
            return atom_name
        else:
            if atom_name in MOL2File.elements_twoLetters:
                return atom_name
            elif atom_name == "CA":
                if sybyl_atom_type == "Ca":
                    return atom_name
                elif sybyl_atom_type in carbon_types:
                    return "C"
                else:
                    raise ValueError(
                        get_error_msg(atom_name, sybyl_atom_type)
                    )
            elif atom_name[:2] in ["CB", "CG", "CD", "CE", "CZ"]:
                return "C"
            elif atom_name[0] == "H" and len(atom_name) > 1:
                return sybyl_atom_type
            elif atom_name[0] == "O":
                return "O"
            else:
                raise ValueError(
                    get_error_msg(atom_name, sybyl_atom_type)
                )

    def __init__(self):
        super().__init__()
        self._mol_name = None
        self._num_atoms = None
        self._num_bonds = None
        self._num_subst = None
        self._num_feat = None
        self._num_sets = None
        self._mol_type = None
        self._charge_type = None
        self._status_bits = None
        self._mol_comment = None
        self._charges = None
        self._ind_molecule = None
        self._ind_atoms = None
        self._ind_bonds = None
        self._sybyl_atom_types = None

    def _get_number_of_molecules(self):
        """
        Calculates the number of molecules from the lines variable and updates
        the according self._ind_molecule variable.
        """
        # check if MOL2File has previous content
        if self._ind_molecule is None:
            self._ind_molecule = [
                    i for i, x in enumerate(
                        self.lines
                    ) if "@<TRIPOS>MOLECULE" in x
            ]
            if len(self._ind_molecule) == 0:
                self._ind_molecule = [0]
        # Only return 0 molecules if only @<TRIPOS>MOLECULE at index 0
        # and also less then 6 lines => no coordinates contained
        cond = self._ind_molecule == [0]
        cond = cond and len(self.lines) <= 6
        N_molecules = 0 if cond else len(self._ind_molecule)
        return N_molecules

    def check_valid_mol2(self):
        self._ind_molecule = [
                i for i, x in enumerate(self.lines) if "@<TRIPOS>MOLECULE" in x
        ]
        if len(self._ind_molecule) == 0:
            raise ValueError(
                "Mol2 File doesn't contain a MOLECULE section, therefore"
                "it is not possibe to parse this file"
            )

        self._ind_atoms = [
                i for i, x in enumerate(self.lines) if "@<TRIPOS>ATOM" in x
        ]
        if len(self._ind_atoms) == 0:
            raise ValueError(
                "Mol2 File doesn't contain a ATOM section, therefore"
                "it is not possibe to parse this file"
            )

        self._ind_bonds = [
                i for i, x in enumerate(self.lines) if "@<TRIPOS>BOND" in x
        ]
        if len(self._ind_bonds) == 0:
            raise ValueError(
                "Mol2 File doesn't contain a BOND section, therefore"
                "it is not possibe to parse this file"
            )

        if len(self._ind_molecule) != len(self._ind_atoms):
            raise ValueError(
                "Mol2 File doesn't contain as many MOLECULE sections as "
                "it does contain ATOM sections"
            )

        if len(self._ind_molecule) != len(self._ind_bonds):
            raise ValueError(
                "Mol2 File doesn't contain as many MOLECULE sections as "
                "it does contain BOND sections"
            )

        if len(self._ind_bonds) != len(self._ind_atoms):
            raise ValueError(
                "Mol2 File doesn't contain as many BOND sections as it does"
                "contain ATOM sections"
            )

    def get_header(self, model=None):
        """
        Get the header from the MOL2 file, if the file contains multiple
        models all headers are parsed and a list is returned for all
        header fields. In this cases a single header can be retrieved by
        specifying the model parameter.

        Parameters
        ----------
        model: int, optional
            Specifies for which of the models contained in this MOL2File
            the according header should be returned.
            If not specified this will return the header for all models,i.e.,
            the according returns will be lists.

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
        # first set all according fields to empty lists
        self._mol_name = []
        self._num_atoms = []
        self._num_bonds = []
        self._num_subst = []
        self._num_feat = []
        self._num_sets = []
        self._mol_type = []
        self._charge_type = []
        self._status_bits = []
        self._mol_comment = []
        # iterate through molecules and read headers based on
        # index of @<TRIPOS>MOLECULE, which is used as offset
        # within the self.lines variable for parsing.
        for ind_mol in self._ind_molecule:
            self._mol_name.append(self.lines[ind_mol+1])
            numbers_line = self.lines[ind_mol+2]
            numbers_parsed = [int(x) for x in numbers_line.strip().split(" ")]
            self._num_atoms.append(numbers_parsed[0])
            num_bonds = -1
            num_subst = -1
            num_feat = -1
            num_sets = -1
            mol_comment = ""
            if len(numbers_parsed) > 1:
                num_bonds = numbers_parsed[1]

            if len(numbers_parsed) > 2:
                num_subst = numbers_parsed[2]

            if len(numbers_parsed) > 3:
                num_feat = numbers_parsed[3]

            if len(numbers_parsed) > 4:
                num_sets = numbers_parsed[4]

            self._num_bonds.append(num_bonds)
            self._num_subst.append(num_bonds)
            self._num_feat.append(num_feat)
            self._num_sets.append(num_sets)

            self._mol_type.append(self.lines[ind_mol + 3])
            self._charge_type.append(self.lines[ind_mol + 4])
            self._status_bits.append(self.lines[ind_mol + 5])
            if "@" not in self.lines[self._ind_molecule[0] + 6]:
                mol_comment = self.lines[self._ind_molecule[0] + 6]
            self._mol_comment.append(mol_comment)

        # if there is only one model in file abuse possibly unset
        # model parameter to get return only the header entries for this
        # However, if model is specified and not 0 in this situation we want
        # to raise the ValueError below.
        if len(self._ind_molecule) == 1 and model is None or model == 0:
            model = 0

        if model is None:
            return (
                self._mol_name, self._num_atoms, self._mol_type,
                self._charge_type, self._num_bonds, self._num_subst,
                self._num_feat, self._num_sets, self._status_bits,
                self._mol_comment
            )
        elif model < len(self._ind_molecule):
            return (
                self._mol_name[model], self._num_atoms[model],
                self._mol_type[model], self._charge_type[model],
                self._num_bonds[model], self._num_subst[model],
                self._num_feat[model], self._num_sets[model],
                self._status_bits[model], self._mol_comment[model]
            )
        else:
            msg = "model = " + str(model) + " is not compatible with file"
            msg += " that has " + str(len(self._ind_molecule)) + " many "
            msg += " models."
            raise ValueError(msg)

    def _write_header(
            self, mol_name, num_atoms, mol_type, charge_type,
            num_bonds=-1, num_subst=-1, num_feat=-1, num_sets=-1,
            status_bits="", mol_comment="", model=0
    ):
        """
        Internal function used to write a single header after
        self.lines has been either filled with as many empty lines
        as needed before or is still filled from a loaded MOL2File.
        """
        N_molecules = len(self._ind_molecule)
        if model >= N_molecules:
            msg = "Can not write header of model = " + str(model)
            msg += " with MOL2File containing " + str(N_molecules)
            msg += " models."
            raise ValueError(msg)
        else:
            mol_index = self._ind_molecule[model]
            if mol_index > len(self.lines):
                msg = "Can not write header to empty file."
                msg += " Use set_structure before set_header."
                msg += "In set_structure a skeleton header will be"
                msg += "generated, which can then be overwritten."
                raise ValueError(msg)

            self.lines[mol_index] = "@<TRIPOS>MOLECULE"
            self.lines[mol_index+1] = mol_name
            line = " " + str(num_atoms)
            if num_bonds >= 0:
                line += " " + str(num_bonds)
                if num_subst >= 0:
                    line += " " + str(num_subst)
                    if num_feat >= 0:
                        line += " " + str(num_feat)
                        if num_sets >= 0:
                            line += " " + str(num_sets)

            self.lines[mol_index+2] = line
            self.lines[mol_index+3] = mol_type
            self.lines[mol_index+4] = charge_type
            self.lines[mol_index+5] = status_bits
            if status_bits != "":
                self.lines[mol_index+6] = mol_comment

    def set_header(
            self, mol_name, num_atoms, mol_type, charge_type,
            num_bonds=-1, num_subst=-1, num_feat=-1, num_sets=-1,
            status_bits="", mol_comment="",
            model=None,
    ):
        """
        Set the header for the MOL2 file according the following structure:

        mol_name
        num_atoms [num_bonds [num_subst [num_feat[num_sets]]]]
        mol_type
        charge_type
        [status_bits
        [mol_comment]]

        For a file containing multiple different structures, the model
        parameter can be used to set the header of the according model. For a
        file containing multiple models of the same Atom, i.e, in the case
        where an AtomArrayStack can be used, the same header is assumed for
        all models. This will be assumed if the MOL2File contains multiple
        models and now model parameter was given.

        taken from
        https://chemicbook.com/2021/02/20/mol2-file-format-explained-for-beginners-part-2.html

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

        model: int, optional
            Specifies for which of the models contained in this MOL2File
            the according header should be set.
            If not specified the header of the first

        """
        # define header for potential output via Exceptions
        header = [
            mol_name, num_atoms, mol_type, charge_type,
            num_bonds, num_subst, num_feat, num_sets,
            status_bits, mol_comment
        ]
        # check if mol_type is valid
        if mol_type != "":
            cond = mol_type in MOL2File.supported_mol_types
            if not cond:
                msg = "The specified molecule type ["
                msg += str(charge_type) + "] \n"
                msg += " is not among the supported molecule types: \n"
                msg += "" + str(MOL2File.supported_mol_types) + "\n"
                msg += "header :: " + str(header) + " \n"
                raise ValueError(msg)

        # check if charge_type is valid
        if charge_type != "":
            cond = charge_type in MOL2File.supported_charge_types
            if not cond:
                msg = "The specified charge type ["
                msg += str(charge_type) + "] "
                msg += " is not among the supported charge types: \n"
                msg += str(MOL2File.supported_charge_types) + "\n"
                raise ValueError(msg)

        N_molecules = self._get_number_of_molecules()
        N_molecules = N_molecules if N_molecules != 0 else 1
        # no model given so simply overwrite member variables with list
        # containing of same data and write said data to all the headers
        if model is None:
            self._mol_name = [mol_name]*N_molecules
            self._num_atoms = [num_atoms]*N_molecules
            self._num_bonds = [num_bonds]*N_molecules
            self._num_subst = [num_subst]*N_molecules
            self._num_feat = [num_feat]*N_molecules
            self._num_sets = [num_sets]*N_molecules
            self._mol_type = [mol_type]*N_molecules
            self._charge_type = [charge_type]*N_molecules
            self._status_bits = [status_bits]*N_molecules
            self._mol_comment = [mol_comment]*N_molecules
            for model in range(N_molecules):
                self._write_header(
                    mol_name, num_atoms, mol_type,
                    charge_type, num_bonds, num_subst,
                    num_feat, num_sets, status_bits,
                    mol_comment, model=model
                )
        elif model < N_molecules:
            self._mol_name[model] = mol_name
            self._num_atoms[model] = num_atoms
            self._num_bonds[model] = num_bonds
            self._num_subst[model] = num_subst
            self._num_feat[model] = num_feat
            self._num_sets[model] = num_sets
            self._mol_type[model] = mol_type
            self._charge_type[model] = charge_type
            self._status_bits[model] = status_bits
            self._mol_comment[model] = mol_comment
            self._write_header(
                mol_name, num_atoms, mol_type,
                charge_type, num_bonds, num_subst,
                num_feat, num_sets, status_bits,
                mol_comment, model=model
            )
        else:
            msg = "Can not write header of model = " + str(model)
            msg += " with MOL2File containing " + str(N_molecules)
            msg += " models."
            raise ValueError(msg)

    def get_structure(self, model=None):
        """
        Get an :class:`AtomArray` from the MOL2 file.

        Parameters
        ----------
        model: int, optional
            If this parameter is set, the according substructure will
            be returned. This can be used to read single structures from
            a multi model MOL2File, containing different molecules.

        Returns
        -------
        array: AtomArray or AtomArrayStack
            This :class:`AtomArray` contains the optional ``charge`` annotation
            and has an associated :class:`BondList`. Furthermore the optional
            ``atom_id`` will be set based upon the first column of the
            MOL2File. If the atom_name column contains only element names,
            only the ``element`` category will be filled. However, if
            the atom_name column contains actual atom_names like, e.g.,
            `CA` for a backbone C-alpha atom, this will be used to set the
            ``atom_name`` category, and according elements will be derived
             from the atom_name and sybyl_atom_type.
        """
        self.get_header()
        N_molecules = self._get_number_of_molecules()
        atom_array_stack = []
        # instantiate atom array and bonds based on number of atoms information
        self._sybyl_atom_types = [[] for i in range(N_molecules)]
        for i in range(N_molecules):
            N_atoms = self._num_atoms[i]
            atoms = AtomArray(N_atoms)
            if self._charge_type != "NO_CHARGES":
                atoms.add_annotation("charge", int)
                if self._charges is None:
                    self._charges = np.zeros(
                        N_atoms
                    ).reshape(
                        (1, N_atoms)
                    )
                else:
                    self._charges = np.vstack(
                        (
                            self._charges,
                            np.zeros(N_atoms)
                        )
                    )

            bonds = BondList(N_atoms)
            # Iterate through all the atom lines by stating from line after
            # @<TRIPOS>ATOM until line starting with @ is reached.
            # All lines in between are assumed to be atom lines and are parsed
            # into atoms accodringly.
            index = self._ind_atoms[i]+1
            j = 0
            atom_type_sybl_row = [""]*N_atoms
            atom_names = []
            while "@" not in self.lines[index]:
                line = [
                    x for x in self.lines[index].strip().split(" ") if x != ''
                ]
                atom_id = int(line[0])
                atom_name = line[1]
                x_coord = float(line[2])
                y_coord = float(line[3])
                z_coord = float(line[4])
                atom_type_sybl = line[5]
                subst_id = -1
                subst_name = ""
                charge = 0.0
                status_bits = ""
                if len(line) > 6:
                    subst_id = int(line[6])

                if len(line) > 7:
                    subst_name = line[7]

                if len(line) > 8:
                    charge = float(line[8])

                if len(line) > 9:
                    status_bits = line[9]

                if self._charge_type != "NO_CHARGES":
                    self._charges[i][j] = charge
                    atom_i = Atom(
                        [x_coord, y_coord, z_coord],
                        charge=int(np.rint(charge))
                    )
                else:
                    atom_i = Atom(
                        [x_coord, y_coord, z_coord],
                    )

                atom_i.atom_id = atom_id
                atom_i.element = atom_name
                atom_names.append(atom_name)
                atom_i.res_id = subst_id
                atom_i.res_name = subst_name
                atoms[j] = atom_i
                atom_type_sybl_row[j] = atom_type_sybl
                index += 1
                j += 1

            # after first teration over structure we need a second pass
            # to correctly infer atom_names and element if necessary from
            # the atom_name and sybyl_atom_type columns
            def filter_func(e):
                cond = len(e) == 1
                cond = cond or e in MOL2File.elements_twoLetters

            is_atom_names_only_element_names = np.all(
                [filter_func(e) for e in atoms.element]
            )

            if not is_atom_names_only_element_names:
                filtered = np.array(
                    [len(x) > 1 for x in atom_names]
                )
                is_atom_name_NOTEQUAL_element = np.any(filtered)

                index = self._ind_atoms[i]+1
                j = 0
                while "@" not in self.lines[index]:
                    atom_j = atoms[j]
                    if is_atom_name_NOTEQUAL_element:
                        atom_name = atom_names[j]
                        element = MOL2File.atom_name_to_element(
                            atom_names[j],
                            atom_type_sybl_row[j]
                        )
                        if self._charge_type[i] != "NO_CHARGES":
                            atoms[j] = Atom(
                                [
                                    atoms[j].coord[0],
                                    atoms[j].coord[1],
                                    atoms[j].coord[2],
                                ],
                                charge=atoms[j].charge,
                                element=element,
                                atom_name=atom_name,
                                res_name=atoms[j].res_name,
                                res_id=atoms[j].res_id
                            )
                        else:
                            atoms[j] = Atom(
                                [
                                    atoms[j].coord[0],
                                    atoms[j].coord[1],
                                    atoms[j].coord[2],
                                ],
                                element=element,
                                atom_name=atom_name,
                                res_name=atoms[j].res_name,
                                res_id=atoms[j].res_id
                            )

                    else:
                        assert len(atom_names[j]) <= 2
                        atoms[j].element = atom_names[j]

                    index += 1
                    j += 1

            # Iterate through all the bond lines by stating from line after
            # @<TRIPOS>BOND until line starting with @ is reached.
            # All lines in between are assumed to be atom lines and are parsed
            # into atoms accodringly.
            index = self._ind_bonds[i] + 1
            while index < len(self.lines) and "@" not in self.lines[index]:
                line = [
                    x for x in self.lines[index].strip().split(" ") if x != ''
                ]
                bond_id = int(line[0])
                origin_atom_id = int(line[1])
                target_atom_id = int(line[2])
                bond_typ = MOL2File.sybyl_to_biotite_bonds[str(line[3])]
                status_bits = ""
                if len(line) > 4:
                    status_bits = str(line[4])

                if bond_typ is not None:
                    bonds.add_bond(
                        origin_atom_id-1,
                        target_atom_id-1,
                        bond_typ
                    )

                index += 1

            atoms.bonds = bonds
            atom_array_stack.append(atoms)
            self._sybyl_atom_types[i].append(atom_type_sybl_row)

        if len(atom_array_stack) == 1:
            return atom_array_stack[0]
        elif model is not None:
            if model < len(atom_array_stack):
                return atom_array_stack[model]
            else:
                msg = "Given model = " + str(model) + " value, incompatible"
                msg += " with MOL2File containing only "
                msg += str(len(atom_array_stack)) + " many models."
                raise ValueError(msg)
        else:
            return struc.stack(atom_array_stack)

    def __append_atom_array(self, atoms, charges=None):
        """
        Internal function that is used to append a single atom
        to the lines member variable.
        """
        n_atoms = atoms.shape[0]
        self.lines.append("@<TRIPOS>ATOM")
        atoms_has_atom_ids = hasattr(atoms, "atom_id")
        if charges is not None:
            assert len(charges) == len(atoms)

        bonds, types = atoms.bonds.get_all_bonds()
        for i, atom in enumerate(atoms):
            atom_id = i+1
            if atoms_has_atom_ids:
                atom_id = atom.atom_id

            line = "{:>7}".format(atom_id)
            if (atoms.atom_name is not None) and (len(atom.atom_name) != 0):
                line += " {:<7}".format(atom.atom_name)
            else:
                line += " {:<7}".format(atom.element)
            line += "{:>11.4f}".format(atom.coord[0])
            line += "{:>10.4f}".format(atom.coord[1])
            line += "{:>10.4f}".format(atom.coord[2])
            line += " {:<8}".format(
                MOL2File.transcribe_sybyl_atom_type(
                    atom, bonds[i], types[i], i
                )
            )
            if atom.res_id != 0:
                line += str(atom.res_id)

                if atom.res_name != "":
                    line += "  " + str(atom.res_name)

                    if self._charge_type != "NO_CHARGES":
                        line += "       "
                        if charges is not None:
                            line += " {: .{}f}".format(charges[i], 4)
                        else:
                            line += " {: .{}f}".format(atom.charge, 4)

            self.lines.append(line)

        self.lines.append("@<TRIPOS>BOND")
        for i, bond in enumerate(atoms.bonds.as_array()):
            line = "{:>6}".format(i+1)
            line += "{:>6}".format(bond[0]+1)
            line += "{:>6}".format(bond[1]+1)
            line += "{:>5}".format(MOL2File.biotite_bonds_to_sybyl[bond[2]])
            self.lines.append(line)

    def set_structure(self, atoms):
        """
        Set the :class:`AtomArray` or :class:`AtomArrayStack` for the file.
        As a remark the heuristic for deriving the sybyl_atom_type is
        currently not complete. It will mostly get it right regarding
        hybridization and aromatic bonds. However, some special cases are
        not covered yet and some two letter elements might produce an
        error as they might not yet be added to the static
        MOL2File.elements_twoLetters array.
        Also, this function doesn't support setting specific substructures in
        an multi model MOL2File based on a model parameter yet.
        This is reflected in self._charges which if previously read from
        an MOL2File needs to be set self._charges[0,:].

        Parameters
        ----------
        array : AtomArray, AtomArrayStack
            The array or stack of arrays to be saved into this file.
            Must have an associated :class:`BondList`.
            Also if the header of this MOL2File has it's charge_type field
            set to something other then 'NO_CHARGES' the AtomArray or
            AtomArrayStack must have an associated charge category.
            Furthermore, if patial_charges have been given via the
            set_charges member function, the charge field will be ignored and
            the set partial_charges will instead be written to the charge
            column of the MOl2File.
        """
        if len(self.lines) < 5:
            def isArrayStack(x):
                return isinstance(x, AtomArrayStack)
            # set skeleton header for file where set_header was not invoked
            self.lines = [""]*6
            self.set_header(
                "",
                atoms.shape[1] if isArrayStack(atoms) else atoms.shape[0],
                "SMALL",
                "NO_CHARGES" if atoms.charge is None else "USER_CHARGES"
            )

        if self._ind_atoms is None:
            # if _ind_atoms is None there was no content before
            # this means a skeleton header has been written and we can set the
            # maximum index consider to the number of lines
            self._ind_atoms = [len(self.lines)]

        # if _ind_atoms was not None this will consider mol_comments
        header_lines = self.lines[:self._ind_atoms[0]]
        # since setting new structure delete all previously stored
        # structure information if any
        self.lines = self.lines[:self._ind_atoms[0]]

        # check if bonds given
        if atoms.bonds is None:
            raise BadStructureError(
                "Input AtomArrayStack has no associated BondList"
            )

        # check charges type
        if self._charge_type != "NO_CHARGES" and atoms.charge is None:
            msg = "Specified charge type "
            msg += str(self._charge_type) + ".\n"
            msg += "but one of given AtomArrays has no charges"
            raise ValueError(msg)

        # if no partial charges set before set_structure
        # use atoms.charge field instead
        if self._charges is None and atoms.charge is not None:
            self._charges = atoms.charge
        elif self._charges is not None and len(self._charges.shape) == 2:
            self._charges = self._charges[0, :]

        # check if number of atoms match with header
        cond1 = isinstance(atoms, AtomArray)
        cond1 = cond1 and (int(self._num_atoms[0]) == int(atoms.shape[0]))
        cond2 = isinstance(atoms, AtomArrayStack)
        cond2 = cond2 and (int(self._num_atoms[0]) == int(atoms.shape[1]))
        if (not cond1) and (not cond2):
            msg = "Mismatch between number of atoms in header :: "
            msg += str(self._num_atoms[0]) + " and number of atoms in given"
            if isinstance(atoms, AtomArray):
                msg += " AtomArray :: " + str(atoms.shape[0]) + " "
            else:
                msg += " AtomArrayStack :: " + str(atoms.shape[1]) + " "
            raise ValueError(msg)

        if isinstance(atoms, AtomArray):
            if self._charges is not None:
                self.__append_atom_array(atoms, self._charges)
            else:
                self.__append_atom_array(atoms)
        elif isinstance(atoms, AtomArrayStack):
            n_models = atoms.shape[0]
            for i, atoms_i in enumerate(atoms):
                # as mentioned the same header will be prepended to all
                # the other models
                if i > 0:
                    self.lines += header_lines

                if self._charges is not None:
                    self.__append_atom_array(atoms_i, self._charges)
                else:
                    self.__append_atom_array(atoms_i)

    def set_charges(self, charges):
        """
        Set the partial charges. This function specifically does not check
        if the ndarray dimension fit with a struture already contained as
        this might make the process of setting charges to a new empty
        MOL2File not containing a structure yet overly complicated.
        It is left to the user to get this right.

        Parameters
        ----------
        charges: ndarray
            A ndarray containing data with `float` type to be written as
            partial charges.
            If ndarray has any other type an according error will be raised.
        """
        if np.issubdtype(charges.dtype, np.floating):
            struct = None
            header = None
            N_molecules = self._get_number_of_molecules()
            if N_molecules != 0:
                header = self.get_header(model=0)
                struct = self.get_structure()

            self._charges = charges
            if N_molecules != 0:
                self.set_structure(struct)
                self.set_header(*header)
        else:
            raise ValueError("Non floating type provided for charges")

    def get_charges(self):
        """
        The getter function for retrieving the partial charges from the read
        MOL2File if it has any. If file with no charges was read and this
        functoin is called a warning will be given.
        Also if the charges member variable is still None this will invoke
        the get_structure function, which in turn might
        raise a BadStructureError if there is no structure contained.

        Returns
        -------
        charges : ndarray, None
            Either ndarray of type `float` or None if no partial_charges
            where contained.
        """
        if self._charge_type == "NO_CHARGES":
            msg = "The read MOl2File had NO_CHARGES set, therefore"
            msg += "no partial charges where contained in the file."
            warning.warn(msg)
            return None

        # If no charges read yet, call get_structure to read
        # structure in, including charges.
        if self._charges is None:
            self.get_structure()

        if len(self._charges) == 1:
            return self._charges[0]
        else:
            return self._charges

    def get_model_count(self):
        """
        Get the number of models contained in the MOL2File.

        Returns
        -------
        model_count : int
            The number of models.
        """
        if self._ind_molecule is None:
            self.check_valid_mol2()
        return len(self._ind_molecule)
