# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from os.path import realpath, dirname, join, isdir
from os import listdir, makedirs
import shutil
from importlib import import_module
import types
import sys
import abc
import inspect


##### API Doc creation #####

_indent = " " * 4
l = []

def create_api_doc(src_path, doc_path):
    if isdir(doc_path):
        shutil.rmtree(doc_path)
    makedirs(doc_path)
    package_list = _create_package_doc("biopython",
                                       join(src_path, "biopython"),
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


##### General #####

package_path = join( dirname(dirname(realpath(__file__))), "src" )
sys.path.insert(0, package_path)
create_api_doc(package_path, "apidoc")

extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.autosummary",
              "sphinx.ext.doctest",
              "sphinx.ext.mathjax",
              "sphinx.ext.viewcode",
              "numpydoc"]

templates_path = ["templates"]
source_suffix = [".rst"]
master_doc = "index"

project = "Biopython"
copyright = "2017, Patrick Kunzmann"
author = "Patrick Kunzmann"
version = "2.0"

exclude_patterns = ["build"]

pygments_style = "sphinx"

todo_include_todos = False


##### HTML #####

html_theme = "alabaster"
html_static_path = ["static"]
#html_logo = "static/assets/general/biopython_logo_xs.png"
html_favicon = "static/assets/general/biopython_icon_32p.png"
htmlhelp_basename = "BiopythonDoc"
html_sidebars = {"**": ["about.html",
                        #"localtoc.html",
                        "navigation.html",
                        "relations.html",
                        "searchbox.html",
                        "donate.html"]}
html_theme_options = {
    "description"      : "A set of general tools "
                         "for computational biology",
    "logo"             : "assets/general/biopython_logo_xs.png",
    "logo_name"        : "false",
    "github_user"      : "padix-key",
    "github_repo"      : "biopython2",
    "github_banner"    : "true",
    "extra_nav_links"  : {"Overview" : "index.html",
                          "Tutorial" : "tutorial/index.html",
                          "API Reference" : "apidoc/biopython.html"},
    "page_width"       : "85%",
    "fixed_sidebar"    : "true"
    
}

##### LaTeX #####

latex_elements = {}
latex_documents = [
    (master_doc, "Biopython.tex", "Biopython Documentation",
     "Patrick Kunzmann", "manual"),
]

