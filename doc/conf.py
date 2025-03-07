# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

# Setup Cython for import of uncompiled *.pyx files
import numpy as np
import pyximport

pyximport.install(
    setup_args={"include_dirs": np.get_include()}, build_in_temp=False, language_level=3
)

import sys
import warnings
from os.path import dirname, join, realpath
import matplotlib
import pybtex
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey
import biotite

BIOTITE_DOMAIN = "www.biotite-python.org"
DOC_PATH = dirname(realpath(__file__))
PACKAGE_PATH = join(dirname(DOC_PATH), "src")


# Include biotite/doc in PYTHONPATH
# in order to import modules for API doc generation etc.
sys.path.insert(0, DOC_PATH)
import apidoc
import bibliography
import preamble
import scraper
import switcher
import viewcode

# Reset matplotlib params
matplotlib.rcdefaults()

# Pregeneration of files
apidoc.create_api_doc(PACKAGE_PATH, join(DOC_PATH, "apidoc"))
switcher.create_switcher_json(join("static", "switcher.json"), "v0.41.0", n_versions=5)

# Use custom citation style
pybtex.plugin.register_plugin("pybtex.style.formatting", "ieee", bibliography.IEEEStyle)

#### Source code link ###

linkcode_resolve = viewcode.linkcode_resolve

#### General ####


# Removed standard matplotlib warning when generating gallery
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a non-GUI backend, "
    "so cannot show the figure.",
)

extensions = [
    "jupyter_sphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "sphinx_copybutton",
    "notfound.extension",
    "numpydoc",
]

templates_path = ["templates"]
source_suffix = [".rst"]
master_doc = "index"

project = "Biotite"
copyright = "The Biotite contributors"
version = biotite.__version__
release = biotite.__version__

exclude_patterns = [
    # These are automatically incorporated by sphinx_gallery
    "examples/scripts/**/README.rst",
    # Execution times are not reported to the user
    "sg_execution_times.rst",
]
# Do not run tutorial code if gallery generation is disabled
if "plot_gallery=0" in sys.argv:
    exclude_patterns.append("tutorial/**/*.rst")

pygments_style = "sphinx"

todo_include_todos = False

# Prevents numpydoc from creating an autosummary which does not work
# properly due to Biotite's import system
numpydoc_show_class_members = False

# Prevent autosummary from using sphinx-autogen, since it would
# overwrite the document structure given by apidoc.json
autosummary_generate = False

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "ieee"

notfound_urls_prefix = "/latest/"

intersphinx_mapping = {
    "rdkit": ("https://www.rdkit.org/docs/", None),
    "openmm": ("http://docs.openmm.org/latest/api-python/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
intersphinx_timeout = 60


#### HTML ####

html_theme = "pydata_sphinx_theme"

html_static_path = ["static"]
html_css_files = ["biotite.css", "fonts.css"]
html_title = "Biotite"
html_logo = "static/assets/general/biotite_logo.svg"
html_favicon = "static/assets/general/biotite_icon_32p.png"
html_baseurl = f"https://{BIOTITE_DOMAIN}/latest/"
html_theme_options = {
    "navbar_start": ["navbar-logo", "version-switcher"],
    "switcher": {
        "json_url": f"https://{BIOTITE_DOMAIN}/latest/_static/switcher.json",
        "version_match": version,
    },
    "show_version_warning_banner": True,
    "header_links_before_dropdown": 7,
    "pygment_light_style": "friendly",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/biotite-dev/biotite",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/biotite/",
            "icon": "fa-solid fa-box-open",
            "type": "fontawesome",
        },
        {
            "name": "News",
            "url": "https://biotite.bsky.social",
            "icon": "fa-brands fa-bluesky",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": True,
    "show_prev_next": False,
    "show_toc_level": 2,
}
html_sidebars = {
    # No primary sidebar for these pages
    "extensions": [],
    "install": [],
    "contribute": [],
    "logo": [],
}
html_context = {
    "github_user": "biotite-dev",
    "github_repo": "biotite",
    "github_version": "master",
    "doc_path": "doc",
}

sphinx_gallery_conf = {
    "examples_dirs": ["examples/scripts/sequence", "examples/scripts/structure"],
    "gallery_dirs": ["examples/gallery/sequence", "examples/gallery/structure"],
    "subsection_order": ExplicitOrder(
        [
            "examples/scripts/sequence/homology",
            "examples/scripts/sequence/sequencing",
            "examples/scripts/sequence/profile",
            "examples/scripts/sequence/annotation",
            "examples/scripts/sequence/misc",
            "examples/scripts/structure/protein",
            "examples/scripts/structure/nucleotide",
            "examples/scripts/structure/molecule",
            "examples/scripts/structure/contacts",
            "examples/scripts/structure/modeling",
            "examples/scripts/structure/alphabet",
            "examples/scripts/structure/misc",
        ]
    ),
    "within_subsection_order": FileNameSortKey,
    # Do not run example scripts with a trailing '_noexec'
    "filename_pattern": "^((?!_noexec).)*$",
    "ignore_pattern": r"(.*ignore\.py)",
    "download_all_examples": False,
    # Never report run time
    "min_reported_time": sys.maxsize,
    "default_thumb_file": join(
        DOC_PATH, "static/assets/general/biotite_icon_thumb.png"
    ),
    "capture_repr": (),
    "image_scrapers": (
        "matplotlib",
        scraper.static_image_scraper,
        scraper.pymol_scraper,
    ),
    "matplotlib_animations": True,
    "image_srcset": ["2x"],
    "backreferences_dir": "examples/backreferences",
    "doc_module": ("biotite",),
    "reset_modules": (preamble.setup_script),
    "remove_config_comments": True,
}


#### App setup ####


def setup(app):
    app.connect("autodoc-skip-member", apidoc.skip_nonrelevant)
