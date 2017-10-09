# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import importlib

def read_structure_from_file(file_name, format=None, extra_fields=[]):
    """
    This methods read a structure from a file.
    
    Parameters
    ----------
    file_name : str
            The name of the file to be read. Includes the file path.
    format : str
            The file format to be used. Can be any of the supported file
            formats as lowercase string. If no argument is given, the format
            will be guessed from the file extension.
    extra_fields : list of str, optional
            The strings in the list are optional annotation categories
            that should be stored in the uoput array or stack.
            There are 4 opional annotation identifiers:
            'atom_id', 'b_factor', 'occupancy' and 'charge'.
        
    Returns
    -------
    array_stack : AtomArrayStack
        A stack containing the annontation arrays from `template`
        but the coordinates from the trajectory file.
        
    Examples
    --------
    Load a `\*.pdb` file:
        >>> from biopython.structure.io import read_structure_from_file
        >>> struct = read_structure_from_file('1l2y.pdb', format='pdb')
    """
    
    # guess file format
    if format is None:
        format = file_name.split('.')[-1]
        
    # load correct reader class
    class_path = "biopython.structure.io.{format}.{prefix}File".format(format=format.lower(), prefix=format.upper())
    module_name, class_name = class_path.rsplit(".", 1)
    MyClass = getattr(importlib.import_module(module_name), class_name)
    reader = MyClass()
    
    # read file and return struct
    reader.read(file_name)
    stack = reader.get_structure(extra_fields)
    return stack
    
