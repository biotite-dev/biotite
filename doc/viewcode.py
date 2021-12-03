# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["linkcode_resolve"]

import sys
from importlib import import_module
from os.path import dirname, join, isdir, splitext
from os import listdir
import inspect


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
    attribute_index : dict( tuple(str, str) -> (str, bool))
        Maps the combination of (sub)package name and attribute to
        the name of a Python module and to a boolean value that
        indicates, whether it is a Cython module.
    cython_line_index : dict( tuple(str, str) -> tuple(int, int) ) )
        Maps the combination of (sub)package name and attribute to
        the first and last line in a Cython module.
        Does not contain entries for attributes that are not part of a
        Cython module.
    """
    if not _is_package(src_path):
        # Directory is not a Python package/subpackage
        # -> Nothing to do
        return {}, {}
    
    attribute_index = {}
    cython_line_index = {}

    # Identify all subdirectories...
    directory_content = listdir(src_path)
    dirs = [f for f in directory_content if isdir(join(src_path, f))]
    # ... and index them recursively 
    for directory in dirs:
        sub_attribute_index, sub_cython_line_index = _index_attributes(
            f"{package_name}.{directory}",
            join(src_path, directory),
        )
        attribute_index.update(sub_attribute_index)
        cython_line_index.update(sub_cython_line_index)
    
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
        module = import_module(module_name)
        is_cython = source_file.endswith(".pyx")
        for attribute in module.__all__:
            attribute_index[(package_name, attribute)] \
                = (module_name, is_cython)
        if is_cython:
            with open(join(src_path, source_file), "r") as cython_file:
                lines = cython_file.read().splitlines()
            for attribute, (first, last) in _index_cython_code(lines).items():
                cython_line_index[(package_name, attribute)] = (first, last)
    
    return attribute_index, cython_line_index


def _index_cython_code(code_lines):
    """
    Find the line position of classes and functions in *Cython* files.

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
    
    Returns
    -------
    line_index : dict (str -> tuple(int, int))
        Maps an attribute name to its first and last line in a Cython
        module.
    """
    line_index = {}

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
        
        line_index[attr_name] = (
            # 'One' based indexing
            attr_line_start + 1,
            # 'One' based indexing and inclusive stop
            attr_line_stop
        )
        
    return line_index


def _is_package(path):
    content = listdir(path)
    return "__init__.py" in content


_attribute_index, _cython_line_index = _index_attributes(
    "biotite",
    # Directory to src/biotite
    join(dirname(dirname(__file__)), "src", "biotite")
)




def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    
    package_name = info["module"]
    attr_name = info["fullname"]
    try:
        module_name, is_cython = _attribute_index[(package_name, attr_name)]
    except KeyError:
        # The attribute is not defined within Biotite
        # It may be e.g. in inherited method from an external source
        return None

    if is_cython:
        if (package_name, attr_name) in _cython_line_index:
            first, last = _cython_line_index[(package_name, attr_name)]
            return f"https://github.com/biotite-dev/biotite/blob/master/src/" \
                   f"{module_name.replace('.', '/')}.pyx#L{first}-L{last}"
        else:
            # In case the attribute is not found
            # by the Cython code analyzer
            return f"https://github.com/biotite-dev/biotite/blob/master/src/" \
                   f"{module_name.replace('.', '/')}.pyx"
    
    else:
        module = import_module(module_name)
        
        # Get the object defined by the attribute name,
        # by traversing the 'attribute tree' to the leaf
        obj = module
        for attr_name_part in attr_name.split("."):
            obj = getattr(obj, attr_name_part)
        
        # Temporarily change the '__module__' attribute, which is set
        # to the subpackage in Biotite, back to the actual module in
        # order to fool Python's inspect module
        obj.__module__ = module_name

        source_lines, first = inspect.getsourcelines(obj)
        last = first + len(source_lines) - 1

        return f"https://github.com/biotite-dev/biotite/blob/master/src/" \
               f"{module_name.replace('.', '/')}.py#L{first}-L{last}"