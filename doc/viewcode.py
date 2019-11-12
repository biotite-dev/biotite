# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["find_actual_module"]

from os.path import dirname, join, isdir, splitext
from os import listdir
from importlib import import_module
import biotite


def _index_attributes(package_name, src_path, attribute_index):
    if not _is_package(src_path):
        # Directory is not a Python package/subpackage
        # -> Nothing to do
        return []
    
    # Identify all subdirectories...
    directory_content = listdir(src_path)
    dirs = [f for f in directory_content if isdir(join(src_path, f))]
    # ... and index them recursively 
    for directory in dirs:
        _index_attributes(
            f"{package_name}.{directory}",
            join(src_path, directory),
            attribute_index
        )
    
    # Import all modules in directory and index attributes
    module_names = [
        f"{package_name}.{splitext(file_name)[0]}"
        for file_name in directory_content
        if file_name != "__init__.py" and (
            # Standard Python modules
            ".py"  in file_name or
            # Extension modules
            ".pyx" in file_name
        )
    ]
    for module_name in module_names:
        module = import_module(module_name)
        for attribute in dir(module):
            attribute_index[(package_name, attribute)] = module_name


def _is_package(path):
    content = listdir(path)
    return "__init__.py" in content


attribute_index = {}
_index_attributes("biotite", dirname(biotite.__file__), attribute_index)


def find_actual_module(app, modname, attribute):
    top_attribute = attribute.split(".")[0]
    return attribute_index[(modname, top_attribute)]