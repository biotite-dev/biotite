# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.sdf"
__author__ = "Benjamin E. Mayer"
__all__ = ["SDFile"]

import datetime
from warnings import warn
import numpy as np
from ...atoms import AtomArray, AtomArrayStack
import biotite.structure as struc
from ....file import TextFile, InvalidFileError
from ...error import BadStructureError
from ..ctab import read_structure_from_ctab, write_structure_to_ctab


# Number of header lines
N_HEADER = 3
DATE_FORMAT = "%d%m%y%H%M"


class SDFile(TextFile):
    """
    This class represents a file in SDF format, that is used to store
    structure information for small molecules. :footcite:`Dalby1992`
    The implementation here is based on the MOLFile and is basically an 
    extension of the MOLFile to als cover multiple models and meta information
    in the form of a dictionary.    
           
    
    References
    ----------
    
    .. footbibliography::
    
    Examples
    --------

    >>> from os.path import join
    >>> mol_file = MOLFile.read(join(path_to_structures, "molecules", "TYR.sdf"))
    >>> atom_array = mol_file.get_structure()
    >>> print(atom_array)
                0             N         1.320    0.952    1.428
                0             C        -0.018    0.429    1.734
                0             C        -0.103    0.094    3.201
                0             O         0.886   -0.254    3.799
                0             C        -0.274   -0.831    0.907
                0             C        -0.189   -0.496   -0.559
                0             C         1.022   -0.589   -1.219
                0             C        -1.324   -0.102   -1.244
                0             C         1.103   -0.282   -2.563
                0             C        -1.247    0.210   -2.587
                0             C        -0.032    0.118   -3.252
                0             O         0.044    0.420   -4.574
                0             O        -1.279    0.184    3.842
                0             H         1.977    0.225    1.669
                0             H         1.365    1.063    0.426
                0             H        -0.767    1.183    1.489
                0             H         0.473   -1.585    1.152
                0             H        -1.268   -1.219    1.134
                0             H         1.905   -0.902   -0.683
                0             H        -2.269   -0.031   -0.727
                0             H         2.049   -0.354   -3.078
                0             H        -2.132    0.523   -3.121
                0             H        -0.123   -0.399   -5.059
                0             H        -1.333   -0.030    4.784
    """
    
    def __init__(self):
        super().__init__()
        # empty header lines
        self.lines = [""] * N_HEADER
    
    def get_header(self):
        """
        Get the header from the SDF file.
        This is identical to the MOL file header as the basic information 
        per model is the same.
        
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
        mol_name        = self.lines[0].strip()
        initials        = self.lines[1][0:2].strip()
        program         = self.lines[1][2:10].strip()
        
        time = None       
        try:
            time        = datetime.datetime.strptime(
                self.lines[1][10:20],
                DATE_FORMAT
            )
        except:
            line = format(self.lines[1][10:20])
            warn("{} could not be interpreted as datetime".format(line))
            
        dimensions      = self.lines[1][20:22].strip()
        scaling_factors = self.lines[1][22:34].strip()
        energy          = self.lines[1][34:46].strip()
        registry_number = self.lines[1][46:52].strip()
        comments        = self.lines[2].strip()
        return mol_name, initials, program, time, dimensions, \
               scaling_factors, energy, registry_number, comments


    def set_header(self, mol_name, initials="", program="", time=None,
                   dimensions="", scaling_factors="", energy="",
                   registry_number="", comments=""):
        """
        Set the header from the SDF file.
        This is identical to the MOL file header as the basic information 
        per model is the same.
        
        Parameters
        ----------
        mol_name : str
            The name of the molecule.
        initials : str, optional
            The author's initials. Maximum length is 2.
        program : str, optional
            The program name. Maximum length is 8.
        time : datetime or date, optional
            The time of file creation.
        dimensions : str, optional
            Dimensional codes. Maximum length is 2.
        scaling_factors : str, optional
            Scaling factors. Maximum length is 12.
        energy : str, optional
            Energy from modeling program. Maximum length is 12.
        registry_number : str, optional
            MDL registry number. Maximum length is 6.
        comments : str, optional
            Additional comments.
        """
        if time is None:
            time = datetime.datetime.now()
        time_str = time.strftime(DATE_FORMAT)

        self.lines[0] = str(mol_name)
        self.lines[1] = (
            f"{initials:>2}"
            f"{program:>8}"
            f"{time_str:>10}"
            f"{dimensions:>2}"
            f"{scaling_factors:>12}"
            f"{energy:>12}"
            f"{registry_number:>6}"
        )
        self.lines[2] = str(comments)


    def get_structure(self):
        """
        Get an :class:`AtomArray` if it only contains one model
        or an :class:'AtomArrayStack' if it contains multiple models.
        
        Returns
        -------
        array : AtomArray, AtomArrayStack
            This :class: contains the optional ``charge``
            annotation and has an associated :class:`BondList`.
            All other annotation categories, except ``element`` are
            empty.
                                    
        """
        


        model_end_lines = [
            i for i in range(len(self.lines)) if '$$$$' == self.lines[i]
        ]
        
        assert(len(model_end_lines) < len(self.lines))
        
        m_end_lines = [
            i for i in range(
                len(self.lines)
            ) if self.lines[i].startswith("M  END")
        ]
        
        # if only $$$$ is forgotten in mol file add it
        if len(model_end_lines) == 0 and len(m_end_lines) !=0:
            self.lines.append("$$$$")
            model_end_lines = [
                i for i in range(len(self.lines)) if '$$$$' == self.lines[i]
            ]
        
        model_start_lines = [0] + [x+1 for x in model_end_lines[:-1]]
        
        if len(model_end_lines) == 0 and len(m_end_lines) == 0:
            msg  = "Trying to get structure from empty file, or"
            msg += "M_END line is missing. \n"
            msg += "Lines where :: \n" + str(self.lines) + "\n\n"
            raise BadStructureError(msg)

        array_stack = []
        
        i_start = 0
        for i in range(len(model_end_lines)):                                  

            ctab_lines = self.lines[
                int(model_start_lines[i]+3):int(m_end_lines[i])
            ]        
                    
            if len(ctab_lines) == 0:
                msg  = "File does not contain structure data"
                msg += "in model " + str(i) + "."
                raise InvalidFileError(msg)
            

            atom_array = read_structure_from_ctab(ctab_lines)                                       
            array_stack.append(atom_array)                        
            

        if len(model_end_lines) == 1:        
            return array_stack[0]
        else:
            return struc.stack(array_stack)
            

    def set_structure(self, atoms):
        """
        Set the :class:`AtomArray` for the file.
        
        Parameters
        ----------
        array : AtomArray, AtomArrayStack
            The array or stack to be saved into this file.
            Must have an associated :class:`BondList`.
        """
        
        header_lines = []
        header_lines.append(self.lines[0])
        header_lines.append(self.lines[1])
        header_lines.append(self.lines[2])
        print("header_lines ::")
        print(header_lines)
        
 
        header_lines = self.lines[:N_HEADER]
        print("header_lines ::")
        print(header_lines)        
        
        
        if isinstance(atoms, AtomArray):
            print(self.lines)
            self.lines = header_lines + write_structure_to_ctab(atoms)
            self.lines += ["$$$$"]
            print(self.lines)

        elif isinstance(atoms, AtomArrayStack):
        
            for i, atoms_i in enumerate(atoms):

                header_lines = self.lines[:N_HEADER]
                print("header_lines ::")
                print(header_lines)       
                
                                       
                if i == 0:
                    self.lines  = header_lines
                else:
                    self.lines += header_lines                    
                print("struct to ctab on structure : atoms_i :: ")
                print(atoms_i)
                print("")
                processed_lines = write_structure_to_ctab(atoms_i)                
                print("|processed_lines| :: " + str(len(processed_lines)))
                print("")                
                print(processed_lines)                
                self.lines += processed_lines                                       
                self.lines += ["$$$$"]
                
        

    def get_metainformation(self):
        """
        Set the :class:`AtomArray` for the file.
        
        Parameters
        ----------        
        annotations: dict
            This dictionary contains all the metainformation given in lines 
            like:
                >  <MODEL>
                1              
        """
    
        meta_info = {}

        model_end_lines = [
            i for i in range(len(self.lines)) if '$$$$' in self.lines[i]
        ]
        m_end_lines = [
            i for i in range(
                len(self.lines)
            ) if self.lines[i].startswith("M  END")
        ]
        model_start_lines = [0] + [x+1 for x in model_end_lines[:-1]]   
        
        
        i_start = 0
        for i in range(len(model_end_lines)):
                        
            sub_meta_info = {}
            
            line_sub_model = self.lines[
                model_start_lines[i]:model_end_lines[i]+1
            ]           
                        
            annotation_line_indices = [
                i for i in range(len(line_sub_model)) if ">" in line_sub_model[i]
            ]


            for j, indx in enumerate(annotation_line_indices):
                annotation_line = line_sub_model[indx]

                if "<" not in annotation_line or ">" not in annotation_line:
                    msg  = "The annotation tag is not defined in the expected"
                    msg += " format <TAGNAME>. Can not parse metainformation."
                    msg += " Line was :: \n\n " +str(annotation_line)                    
                    raise BadStructureError(msg)                     
                
                annotation_line = annotation_line.strip().strip(">")
                annotation_tag = annotation_line.strip().strip("<").strip()              
                                                                                      
                

                annotation_content = ""
                if j < len(annotation_line_indices)-1:
                    annotation_content = line_sub_model[int(indx+1):annotation_line_indices[j+1]]
                else:          
                    annotation_content = line_sub_model[int(indx+1):]
                    
                line_filter = lambda x: len(x) != 0 and "$$$$" not in x                    
                annotation_content = [
                    x for x in annotation_content if line_filter(x)
                ]
                annotation_content = '\n'.join(annotation_content)                                        
                sub_meta_info[annotation_tag] = annotation_content                        
                                 
            
            meta_info["model_"+str(i)]=sub_meta_info               

        if len(list(meta_info.keys()))==1:
            return meta_info["model_0"]
        else:
            return meta_info     
            
    def set_metainformation(self, meta_information):        
        """
        Set the :class:`AtomArray` for the file.
        
        Parameters
        ----------        
        annotations: dict
            This dictionary contains all the metainformation given in lines 
            like:
                >  <MODEL>
                1  
            Either a single dictionary with MODEL:value pairs or a dictionary 
            of dictionary where each key indicates a model and the value is the 
            according model specific dictionary:
            
            {
                "model_0": {"Tag1":Value1,.... }, 
                "model_1": {"Tag1":Value1',.... }, 
                ...
            }                
        """
                   
    
        header_lines = self.lines[:N_HEADER]
        
        model_end_lines = [
            i for i in range(len(self.lines)) if '$$$$' in self.lines[i]
        ]        
        
        if len(model_end_lines) == 1:
            
            if meta_information is not None:
                for x in meta_information.keys():
                    self.lines.append("> <" +str(x) + "> ")
                    self.lines.append(meta_information[x])

        else:            
            if meta_information is not None:                                                        
                for key in meta_information.keys():
                    sub_annotations = meta_information[key]
                    for subkey in meta_information[key]:                
                        self.lines.append("> <" +str(subkey) + "> ")
                        self.lines.append(sub_annotations[subkey])   
                        

                


def _get_ctab_lines(lines):
    for i, line in enumerate(lines):
        if line.startswith("M  END"):
            return lines[N_HEADER:i+1]
    return lines[N_HEADER:]
