# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.sdf"
__author__ = "Benjamin E. Mayer"
__all__ = [
    "get_structure", "set_structure", 
    "get_header", "set_header",     
    "get_metainformation", "set_metainformation"
]

def get_header(sdf_file):
    """
    """
    return sdf_file.get_header()
    

def set_header(sdf_file, mol_name, initials="", program="", time=None,
                   dimensions="", scaling_factors="", energy="",
                   registry_number="", comments=""):
    """
    sdf
    """
    sdf_file.set_header(
        mol_name, initials="", program="", time=None,
        dimensions="", scaling_factors="", energy="",
        registry_number="", comments=""    
    )    

def get_structure(sdf_file):
    """
    Get an :class:`AtomArray` from the MOL file.

    This function is a thin wrapper around
    :meth:`SDFile.get_structure()`.

    Parameters
    ----------
    sdf_file : SDFile
        The MOL file.
    
    Returns
    -------
    array : AtomArray, AtomArrayStack
        This :class:`AtomArray` contains the optional ``charge``
        annotation and has an associated :class:`BondList`.
        All other annotation categories, except ``element`` are
        empty.
    """
    return sdf_file.get_structure()
    

def set_structure(sdf_file, atoms):
    """
    Set the :class:`AtomArray` for the MOL file.

    This function is a thin wrapper around
    :meth:`SDFile.set_structure()`.
    
    Parameters
    ----------
    sdf_file : SDFile
        The MOL file.
    array : AtomArray
        The array to be saved into this file.
        Must have an associated :class:`BondList`.
    """
    sdf_file.set_structure(atoms)
    
    
def get_metainformation(sdf_file):
    """
    Get the meta informatoin dictionary for the class SDFile.

    This function is a thin wrapper around
    :meth:`SDFile.get_metainformation()`.
    
    Returns
    -------
    meta_information: dict
        This annotationa dictionary contains all the metainformation from
        the SD file parsed as key:value pairs with the annotation tag
        defined via '> <TAG>' being the key and the lines until the next
        annotation tag or end of file being the value.
        If a multi model file has been loaded a dictionary of such dictionaries
        is returned respectively. With the sub-dictionaries containing the
        read meta inforamtion of the sub models.
    
    """
    return sdf_file.get_metainformation()
    
def set_metainformation(sdf_file, meta_information):
    """
    Set the :class:`AtomArray` for the MOL file.

    This function is a thin wrapper around
    :meth:`SDFile.set_structure(meta_info)`.
    
    Parameters
    ----------
    meta_information: dict
        For a single model SDFile a simple dictionary is expected.
        Otherwise a dictionary of dictionaries is expected having as many
        sub dictionaries as there are models in the SD File.
    
    """
    sdf_file.set_metainformation(meta_information)            
