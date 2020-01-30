# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs': np.get_include()})

from os.path import realpath, dirname, join, basename
from os import listdir, makedirs
import sys
import glob
import shutil
import types
import matplotlib

doc_path = dirname(realpath(__file__))

# Include biotite/src in PYTHONPATH
# in order to import the 'biotite' package
package_path = join(dirname(doc_path), "src")
sys.path.insert(0, package_path)
import biotite

# Include biotite/doc in PYTHONPATH
# in order to import modules for API doc generation etc.
sys.path.insert(0, doc_path)
import apidoc
import viewcode
import tutorial
import scraper


#Reset matplotlib params
matplotlib.rcdefaults()

# Creation of API documentation
apidoc.create_api_doc(package_path, join(doc_path, "apidoc"))

# Creation of tutorial *.rst files from Python script
tutorial.create_tutorial("tutorial_src", "tutorial")


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
copyright = "2017-2019, the Biotite contributors"
version = biotite.__version__

exclude_patterns = ["build"]

pygments_style = "sphinx"

todo_include_todos = False

# Prevents numpydoc from creating an autosummary which does not work
# properly due to Biotite's import system
numpydoc_show_class_members = False

autodoc_member_order = "bysource"


#### HTML ####

html_theme = "alabaster"
html_static_path = ["static"]
html_css_files = [
    "biotite.css",
    "https://fonts.googleapis.com/css?" \
        "family=Crete+Round|Fira+Sans|&display=swap",
]
html_favicon = "static/assets/general/biotite_icon_32p.png"
htmlhelp_basename = "BiotiteDoc"
html_sidebars = {"**": ["about.html",
                        "navigation.html",
                        "searchbox.html",
                        "buttons.html"]}
html_theme_options = {
    "description"   : "A comprehensive library for " \
                      "computational molecular biology",
    "logo"          : "assets/general/biotite_logo_s.png",
    "logo_name"     : "false",
    "github_user"   : "biotite-dev",
    "github_repo"   : "biotite",
    "github_banner" : "true",
    "github_button" : "true",
    "github_type"   : "star",
    "page_width"    : "1200px",
    "fixed_sidebar" : "true",
    
    "sidebar_link_underscore" : "#FFFFFF"
}

sphinx_gallery_conf = {
    "examples_dirs"             : "examples/scripts",
    "gallery_dirs"              : "examples/gallery",
    # Do not run example scripts with a trailing '_noexec'
    "filename_pattern"          : "^((?!_noexec).)*$",
    "ignore_pattern"            : ".*ignore\.py",
    "backreferences_dir"        : False,
    "download_section_examples" : False,
    # Never report run time
    "min_reported_time"         : sys.maxsize,
    "default_thumb_file"        : join(
        doc_path, "static/assets/general/biotite_icon_thumb.png"
    ),
    "image_scrapers": ("matplotlib", scraper.static_image_scraper),
}


#### App setup ####

def setup(app):
    app.connect("autodoc-skip-member", apidoc.skip_non_methods)
    app.connect("viewcode-follow-imported", viewcode.find_actual_module)
    app.connect("viewcode-find-source", viewcode.index_source)