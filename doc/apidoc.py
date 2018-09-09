# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["create_api_doc", "skip_non_methods"]

from os.path import dirname, join, isdir
from os import listdir, makedirs
from importlib import import_module
import types
import json


_indent = " " * 3


#with open("apidoc.json", "r") as file:
#    content_structure = json.load(f)



def create_api_doc(src_path, doc_path):
    """
    Create *.rst files for API documentation.

    Parameters
    ----------
    src_path : str
        The path to the working copy of the *Biotite* package
    src_path : str
        The path to the documentation root directory (``biotite/doc``)
    """
    package_list = _create_package_doc("biotite",
                                       join(src_path, "biotite"),
                                       doc_path)
    _create_package_index(doc_path, package_list)


def _create_package_doc(pck, src_path, doc_path):
    if not _is_package(src_path):
        # Directory is not a Python package/subpackage
        # -> Nothing to do
        return []
    # Identify all subdirectories...
    content = listdir(src_path)
    dirs = [f for f in content if isdir(join(src_path, f))]
    # ... and recursively create also the documentation for them 
    sub_pck = []
    for directory in dirs:
        sub_pck += _create_package_doc(
            f"{pck}.{directory}", join(src_path, directory), doc_path
        )
    
    # Import package (__init__.py) and find all attribute names
    module = import_module(pck)
    attr_list = dir(module)
    # Classify attribute names into classes and functions
    func_list = [attr for attr in attr_list
                    if attr[0] != "_"
                    and type(getattr(module, attr))
                    in [types.FunctionType, types.BuiltinFunctionType]
                ]
    class_list = [attr for attr in attr_list
                    if attr[0] != "_"
                    and isinstance(getattr(module, attr), type)]
    # Create directory to store *.rst files for this package/subpackage
    pck_path = join(doc_path, pck)
    if not isdir(pck_path):
        makedirs(pck_path)
    # Create *.rst files
    _create_package_page(pck_path, pck, class_list, func_list, sub_pck)
    for class_name in class_list:
        _create_class_page(pck_path, pck, class_name)
    for function_name in func_list:
        _create_function_page(pck_path, pck, function_name)
    
    return([pck] + sub_pck)


def _create_package_page(doc_path, package_name,
                         classes, functions, subpackages):
    # String for class enumeration
    classes_string = "\n".join(
        [_indent + f"- :doc:`{package_name}.{cls} <{cls}>`"
         for cls in classes]
    )

    # String for function enumeration
    functions_string = "\n".join(
        [_indent + f"- :doc:`{package_name}.{func} <{func}>`"
         for func in functions]
    )

    # String for subpackage enumeration
    subpackages_string = "\n".join(
        [_indent + f"- :doc:`{pck} <../{pck}/package>`"
         for pck in subpackages]
    )

    # Assemble page
    file_content = \
f"""
{package_name}
{"=" * len(package_name)}
.. currentmodule:: matplotlib.axes

.. automodule:: {package_name}

Classes
-------

{classes_string}

Functions
---------

{functions_string}

Subpackages
-----------

{subpackages_string}
"""
    with open(join(doc_path, "package.rst"), "w") as f:
        f.write(file_content)


def _create_class_page(doc_path, package_name, class_name):
    file_content = \
f"""
:orphan:

{package_name}.{class_name}
{"=" * (len(package_name)+len(class_name)+1)}
.. autoclass:: {package_name}.{class_name}
    :show-inheritance:
    :members:
    :undoc-members:
    :inherited-members:
"""
    with open(join(doc_path, f"{class_name}.rst"), "w") as f:
        f.write(file_content)


def _create_function_page(doc_path, package_name, function_name):
    file_content = \
f"""
:orphan:

{package_name}.{function_name}
{"=" * (len(package_name)+len(function_name)+1)}
.. autofunction:: {package_name}.{function_name}
"""
    with open(join(doc_path, f"{function_name}.rst"), "w") as f:
        f.write(file_content)


def _create_package_index(doc_path, package_list):
    lines = []
    
    lines.append("API Reference")
    lines.append("=" * len("API Reference"))
    lines.append("\n")
    
    lines.append(".. toctree::")
    lines.append(_indent + ":maxdepth: 1")
    lines.append("\n")
    for pck in package_list:
        lines.append(_indent + f"{pck}/package")
    with open(join(doc_path, "index.rst"), "w") as f:
        f.writelines([line+"\n" for line in lines])


def _is_package(path):
    content = listdir(path)
    return "__init__.py" in content



# Skip all class members, that are not methods,
# since other attributes are already documented in the class docstring
def skip_non_methods(app, what, name, obj, skip, options):
    """
    Skip all class members, that are not methods,
    since other attributes are already documented
    in the class docstring.
    """
    if what == "class":
        if type(obj) not in [types.FunctionType, types.BuiltinFunctionType]:
            return True