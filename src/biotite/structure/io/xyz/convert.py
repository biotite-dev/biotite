# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.xyz"
__author__ = "Benjamin E. Mayer"
__all__ = ["get_structure", "set_structure", "get_model_count"]



def get_structure(xyz_file, model=None):
    """
    Get an :class:`AtomArray` from the XYZ File.

    This function is a thin wrapper around
    :meth:`XYZFile.get_structure(xyz_file, model)`.

    Parameters
    ----------
    xyz_file : XYZFile
        The XYZ File.
    
    Returns
    -------
    array : AtomArray, AtomArrayStack
        The return type depends on the `model` parameter.
        In either case if only a single model is contained in the ``xyz_file``,
        or the model parameter has been given, a class:`AtomArray` object 
        will be returned. This :class:`AtomArray` only contains the ``element``
        annotation category, all other annotations are empty, as only
        element type and coordinates are contained for the atoms in an 
        *.xyz file.
        Respectively if no model parameter has been specified and the 
        ``xyz_file`` contains multiple models a class:`AtomArrayStack` 
        object will be returned. Of course this class:`AtomArrayStack` 
        object also will only contain the ``element`` category.
    """
    return xyz_file.get_structure(model)
    

def set_structure(xyz_file, atoms):
    """
    Set the :class:`AtomArray` for the XYZ File
    or :class:`AtomArrayStack for a XYZ File with multiple states.

    This function is a thin wrapper around
    :meth:`XYZFile.set_structure()`.
    
    Parameters
    ----------
    xyz_file : XYZFile
        The XYZ File.
    array : AtomArray or AtomArrayStack
        The array to be saved into this file. If a stack is given, each array in
        the stack will be in a separate model and the names of the 
        model will be set to an index enumerating them.
    """
    xyz_file.set_structure(atoms)
    
    
def get_model_count(xyz_file):
    """
    Get the number of models contained in the xyz file.
    
    This function is a thin wrapper around
    :meth:`XYZFile.get_model_count()`.    

    Returns
    -------
    model_count : int
        The number of models.
    """
    return xyz_file.get_model_count()
   
