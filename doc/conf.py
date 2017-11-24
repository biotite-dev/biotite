# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

from os.path import realpath, dirname, join, isdir
from os import listdir, makedirs
import shutil
from importlib import import_module
import types
import sys
import abc
import inspect

package_path = join( dirname(dirname(realpath(__file__))), "src" )
sys.path.insert(0, package_path)
import biotite


##### API Doc creation #####

_indent = " " * 4
l = []

def create_api_doc(src_path, doc_path):
    if isdir(doc_path):
        shutil.rmtree(doc_path)
    makedirs(doc_path)
    package_list = _create_package_doc("biotite",
                                       join(src_path, "biotite"),
                                       doc_path)
    create_package_index(doc_path, package_list)


def _create_package_doc(pck, src_path, doc_path):
    if not _is_package(src_path):
        return []
    else:
        content = listdir(src_path)
        dirs = [f for f in content if isdir(join(src_path, f))]
        sub_pck = []
        for directory in dirs:
            sub = _create_package_doc(pck + "." + directory,
                                      join(src_path, directory),
                                      doc_path)
            sub_pck += sub
        
        module = import_module(pck)
        attr_list = dir(module)
        func_list = [attr for attr in attr_list
                     if attr[0] != "_"
                     and type(getattr(module, attr)) == types.FunctionType]
        class_list = [attr for attr in attr_list
                     if attr[0] != "_"
                     and type(getattr(module, attr)) in [type, abc.ABCMeta]]
        class_dict = {}
        for cls in class_list:
            cls_attr_list = dir(getattr(module, cls))
            class_dict[cls] = [attr for attr in cls_attr_list
                               if attr[0] != "_"
                               and inspect.isfunction(
                                        getattr(getattr(module, cls), attr))
                              ]
        _create_files(doc_path, pck, class_dict, func_list, sub_pck)
        
        return([pck] + sub_pck)


def _create_files(doc_path, package, classes, functions, subpackages):
    sub_path = join(doc_path, package)
    if not isdir(sub_path):
        makedirs(sub_path)
    
    for cls, methods in classes.items():
        file_content = \
        """
:orphan:

{:}.{:}
{:}

.. autodata:: {:}.{:}

|
        """.format(package, cls, "=" * (len(package)+len(cls)+1),
                   package, cls)
        for method in methods:
            file_content += \
            """
.. automethod:: {:}.{:}.{:}

|
            """.format(package, cls, method)
        with open(join(sub_path, cls+".rst"), "w") as f:
            f.write(file_content)
            
    for func in functions:
        file_content = \
        """
:orphan:

{:}.{:}
{:}

.. autofunction:: {:}.{:}
        """.format(package, func, "=" * (len(package)+len(func)+1),
                   package, func)
        with open(join(sub_path, func+".rst"), "w") as f:
            f.write(file_content)
    
    
    lines = []
    
    lines.append(package)
    lines.append("=" * len(package))
    lines.append("\n")
    lines.append(".. automodule:: " + package)
    lines.append("\n")
    
    lines.append("Classes")
    lines.append("-" * len("Classes"))
    lines.append("\n")
    for cls in classes:
        lines.append(_indent + "- :doc:`"
                     + package + "." + cls
                     + " <" + package + "/" + cls + ">`")
    lines.append("\n")
    
    lines.append("Functions")
    lines.append("-" * len("Functions"))
    lines.append("\n")
    for func in functions:
        lines.append(_indent + "- :doc:`"
                     + package + "." + func
                     + " <" + package + "/" + func + ">`")
    lines.append("\n")
    
    lines.append("Subpackages")
    lines.append("-" * len("Subpackages"))
    lines.append("\n")
    for pck in subpackages:
        lines.append(_indent + "- :doc:`"
                     + pck
                     + " <" + pck + ">`")
    lines.append("\n")
    
    with open(join(doc_path, package+".rst"), "w") as f:
        f.writelines([line+"\n" for line in lines])
        
        
def create_package_index(doc_path, package_list):
    
    lines = []
    
    lines.append("API Reference")
    lines.append("-" * len("API Reference"))
    lines.append("\n")
    
    lines.append(".. toctree::")
    lines.append(_indent + ":maxdepth: 1")
    lines.append("\n")
    for pck in package_list:
        lines.append(_indent + pck)
    with open(join(doc_path, "index"+".rst"), "w") as f:
        f.writelines([line+"\n" for line in lines])
    

def _is_package(path):
    content = listdir(path)
    return "__init__.py" in content


create_api_doc(package_path, "apidoc")

##### General #####

extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.autosummary",
              "sphinx.ext.doctest",
              "sphinx.ext.mathjax",
              "sphinx.ext.viewcode",
              "numpydoc"]

templates_path = ["templates"]
source_suffix = [".rst"]
master_doc = "index"

project = "Biotite"
copyright = "2017, the Biotite contributors"
version = biotite.__version__

exclude_patterns = ["build"]

pygments_style = "sphinx"

todo_include_todos = False


##### HTML #####

html_theme = "alabaster"
html_static_path = ["static"]
html_favicon = "static/assets/general/biotite_icon_32p.png"
htmlhelp_basename = "BiotiteDoc"
html_sidebars = {"**": ["about.html",
                        #"localtoc.html",
                        "navigation.html",
                        "relations.html",
                        "searchbox.html",
                        "donate.html"]}
html_theme_options = {
    "description"      : "A general framework for computational biology",
    "logo"             : "assets/general/biotite_logo_s.png",
    "logo_name"        : "false",
    "github_user"      : "biotite-dev",
    "github_repo"      : "biotite",
    "github_banner"    : "true",
    "page_width"       : "85%",
    "fixed_sidebar"    : "true"
    
}

