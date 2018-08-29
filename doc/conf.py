# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs': np.get_include()})

from os.path import realpath, dirname, join, isdir, isfile, basename
from os import listdir, makedirs
import sys
import glob
import shutil
import matplotlib
from importlib import import_module
import types
import abc

absolute_path = dirname(realpath(__file__))
package_path = join(dirname(absolute_path), "src")
sys.path.insert(0, package_path)
import biotite

### Create API docmentation ###

_indent = " " * 3

def create_api_doc(src_path, doc_path):
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
                     and type(getattr(module, attr))
                     in [types.FunctionType, types.BuiltinFunctionType]
                    ]
        class_list = [attr for attr in attr_list
                     if attr[0] != "_"
                     and isinstance(getattr(module, attr), type)]
        _create_files(doc_path, pck, class_list, func_list, sub_pck)
        
        return([pck] + sub_pck)


def _create_files(doc_path, package, classes, functions, subpackages):
    sub_path = join(doc_path, package)
    if not isdir(sub_path):
        makedirs(sub_path)
    
    for cls in classes:
        file_content = \
        """
:orphan:

{:}.{:}
{:}
.. autoclass:: {:}.{:}
    :show-inheritance:
    :members:
    :undoc-members:
    :inherited-members:
        """.format(package, cls, "=" * (len(package)+len(cls)+1),
                   package, cls)
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
    lines.append("=" * len("API Reference"))
    lines.append("\n")
    
    lines.append(".. toctree::")
    lines.append(_indent + ":maxdepth: 1")
    lines.append("\n")
    for pck in package_list:
        lines.append(_indent + pck)
    with open(join(doc_path, "index.rst"), "w") as f:
        f.writelines([line+"\n" for line in lines])


def _is_package(path):
    content = listdir(path)
    return "__init__.py" in content

### Reset matplotlib params ###

matplotlib.rcdefaults()

### Creation of API documentation ###

create_api_doc(package_path, join(absolute_path, "apidoc"))

#### General ####

extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.autosummary",
              "sphinx.ext.doctest",
              "sphinx.ext.mathjax",
              "sphinx.ext.viewcode",
              "sphinx_gallery.gen_gallery",
              "numpydoc"]

templates_path = ["templates"]
source_suffix = [".rst"]
master_doc = "index"

project = "Biotite"
copyright = "2017-2018, the Biotite contributors"
version = biotite.__version__

exclude_patterns = ["build"]

pygments_style = "sphinx"

todo_include_todos = False

# Prevents numpydoc from creating an autosummary which does not work
# due to Biotite's import system
numpydoc_show_class_members = False


#### HTML ####

html_theme = "alabaster"
html_static_path = ["static"]
html_favicon = "static/assets/general/biotite_icon_32p.png"
htmlhelp_basename = "BiotiteDoc"
html_sidebars = {"**": ["about.html",
                        "navigation.html",
                        "searchbox.html",
                        "buttons.html"]}
html_theme_options = {
    "description"   : "A comprehensive framework for " \
                      "computational molecular biology",
    "logo"          : "assets/general/biotite_logo_s.png",
    "logo_name"     : "false",
    "github_user"   : "biotite-dev",
    "github_repo"   : "biotite",
    "github_banner" : "true",
    "page_width"    : "85%",
    "fixed_sidebar" : "true",
    
    "sidebar_link_underscore" : "#FFFFFF"
}

sphinx_gallery_conf = {
    "examples_dirs"             : "examples/scripts",
    "gallery_dirs"              : "examples/gallery",
    "filename_pattern"          : "/",
    "backreferences_dir"        : False,
    "download_section_examples" : False,
    # Never report run time
    "min_reported_time"         : sys.maxsize,
    "default_thumb_file"        : join(
        absolute_path, "static/assets/general/biotite_icon_thumb.png"
    )
}