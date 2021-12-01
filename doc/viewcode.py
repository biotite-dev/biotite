# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["find_actual_module", "index_source"]

from os.path import dirname, join, isdir, splitext
from os import listdir
from importlib import import_module
import biotite


def _index_attributes(package_name, src_path):
    """
    Assign a Python module to each combination of (sub)package and
    attribute (e.g. function, class, etc.) in a given (sub)package.

    Parameters
    ----------
    package_name : str
        Name of the (sub)package.
    src_path : str
        File path to `package_name`.
    
    Parameters
    ----------
    attribute_index : dict( tuple(str, str) -> str)
        Maps the combintation of (sub)package name and attribute to
        the name of a Python module.
    file_index : dict( str -> str )
        Maps a Python module to its file path.
    """
    attribute_index = {}
    file_index = {}
    
    if not _is_package(src_path):
        # Directory is not a Python package/subpackage
        # -> Nothing to do
        return attribute_index, file_index
    
    # Identify all subdirectories...
    directory_content = listdir(src_path)
    dirs = [f for f in directory_content if isdir(join(src_path, f))]
    # ... and index them recursively 
    for directory in dirs:
        sub_attribute_index, sub_file_index = _index_attributes(
            f"{package_name}.{directory}",
            join(src_path, directory),
        )
        attribute_index.update(sub_attribute_index)
        file_index.update(sub_file_index)
    
    # Import all modules in directory and index attributes
    source_files = [
        file_name for file_name in directory_content
        if file_name != "__init__.py" and (
            # Standard Python modules
            file_name.endswith(".py") or
            # Extension modules
            file_name.endswith(".pyx")
        )
    ]
    for source_file in source_files:
        module_name = f"{package_name}.{splitext(source_file)[0]}"
        file_index[module_name] = join(src_path, source_file)
        module = import_module(module_name)
        for attribute in module.__all__:
            attribute_index[(package_name, attribute)] = module_name
    
    return attribute_index, file_index


def _is_package(path):
    content = listdir(path)
    return "__init__.py" in content


_attribute_index, _file_index = _index_attributes(
    "biotite",
    # Directory to src/biotite
    join(dirname(dirname(__file__)), "src", "biotite")
)




def find_actual_module(app, modname, attribute):
    top_attribute = attribute.split(".")[0]
    return _attribute_index[(modname, top_attribute)]


def index_source(app, modname):
    source_file = _file_index[modname]
    
    if source_file.endswith(".pyx"):
        with open(source_file) as file:
            code = file.read()
        code_lines = code.split("\n")
        tags = _analyze_cython_code(code_lines)
        return code, tags
        
    else:
        # For normal Python files return None to tell Sphinx viewcode
        # to do the default handling for these files 
        return None


def _analyze_cython_code(code_lines):
    """
    Find the position of classes and functions in *Cython* files.

    This analyzer works in a very simple way:
    It looks for the `def` and `class` keywords at zero-indentation
    level and determines the end of a class/function by the start of the
    next zero-indentation level attribute or the end of the file,
    respectively.

    By the nature of this approach, methods or inner classes are not
    identified.
    
    Parameters
    ----------
    code_lines : list of str
        The *Cython* source code splitted into lines.
    indent : int
        Attributes are The indentation level.
    
    Returns
    -------
    tags : dict
        The tag dictionary, as expected by the *Sphinx*
        ``viewcode-find-source`` event.
    """
    tags = {}

    for i in range(len(code_lines)):
        line = code_lines[i]
        stripped_line = line.strip()
        
        # Skip empty and comment lines
        if len(stripped_line) == 0 or stripped_line[0] == "#":
            continue
        
        if line.startswith(("def")):
            attr_type = "def"
            # Get name of the function:
            # Remove 'def' from line...
            cropped_line = stripped_line[3:].strip()
            # ...and determine the end of the name by finding the
            # subsequent '('
            cropped_line = cropped_line[:cropped_line.index("(")].strip()
            attr_name = cropped_line
        elif line.startswith(("class", "cdef class")):
            attr_type = "class"
            cropped_line = stripped_line
            # Get name of the class:
            # Remove potential 'cdef' from line...
            if cropped_line.startswith("cdef"):
                cropped_line = cropped_line[4:].strip()
            # ...and remove 'class' from line...
            cropped_line = cropped_line[5:].strip()
            # ...and determine the end of the name by finding the
            # subsequent '(' or ':'
            index = cropped_line.index("(") if "(" in cropped_line \
                    else cropped_line.index(":")
            cropped_line = cropped_line[:index].strip()
            attr_name = cropped_line
        else:
            # No new attribute -> skip line
            continue

        attr_line_start = i
        attr_line_stop = i+1
        for j in range(i+1, len(code_lines)):
            attr_line = code_lines[j]
            if len(attr_line.strip()) == 0 or attr_line.strip()[0] == "#":
                continue
            indent = len(attr_line) - len(attr_line.lstrip())
            if indent == 0:
                # No indentation -> end of attribute
                break
            else:
                # Exclusive stop -> +1
                attr_line_stop = j + 1
        
        tags[attr_name] = (
            attr_type,
            # 'One' based indexing
            attr_line_start + 1,
            # 'One' based indexing and inclusive stop
            attr_line_stop
        )
        
    return tags