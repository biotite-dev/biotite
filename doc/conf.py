# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

# Setup Cython for import of uncompiled *.pyx files
import pyximport
import numpy as np
pyximport.install(
    setup_args={'include_dirs': np.get_include()},
    build_in_temp=False,
    language_level=3
)

from os.path import realpath, dirname, join, basename
import sys
import types
import warnings
from sphinx_gallery.sorting import FileNameSortKey
import matplotlib

import biotite

doc_path = dirname(realpath(__file__))
package_path = join(dirname(doc_path), "src")

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
if not "plot_gallery=0" in sys.argv:
    tutorial.create_tutorial(
        join("tutorial", "src"),
        join("tutorial", "target")
    )


#### General ####

import warnings

# Removed standard matplotlib warning when generating gallery
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a non-GUI backend, "
            "so cannot show the figure."
)

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
copyright = "2017-2020, the Biotite contributors"
version = biotite.__version__
release = biotite.__version__

exclude_patterns = ["build"]

pygments_style = "sphinx"

todo_include_todos = False

# Prevents numpydoc from creating an autosummary which does not work
# properly due to Biotite's import system
numpydoc_show_class_members = False

# Prevent autosummary from using sphinx-autogen, since it would
# overwrite the document structure given by apidoc.json
autosummary_generate = False

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
    
    "sidebar_link_underscore" : "#FFFFFF",
    "link"                    : "#006B99",
}

sphinx_gallery_conf = {
    "examples_dirs"             : "examples/scripts",
    "gallery_dirs"              : "examples/gallery",
    "within_subsection_order"   : FileNameSortKey,
    # Do not run example scripts with a trailing '_noexec'
    "filename_pattern"          : "^((?!_noexec).)*$",
    "ignore_pattern"            : "(.*ignore\.py)|(.*pymol\.py)",
    "backreferences_dir"        : None,
    "download_all_examples" : False,
    # Never report run time
    "min_reported_time"         : sys.maxsize,
    "default_thumb_file"        : join(
        doc_path, "static/assets/general/biotite_icon_thumb.png"
    ),
    "image_scrapers"            : (
        "matplotlib",
        scraper.static_image_scraper,
        scraper.pymol_scraper
    ),
    "matplotlib_animations"     : True,
    "backreferences_dir"        : "examples/backreferences",
    "doc_module"                : ("biotite",),
}


#### App setup ####

def setup(app):
    app.connect("autodoc-skip-member", apidoc.skip_non_methods)
    app.connect("viewcode-follow-imported", viewcode.find_actual_module)
    app.connect("viewcode-find-source", viewcode.index_source)