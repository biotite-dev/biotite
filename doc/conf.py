# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from os.path import realpath, dirname, join
import sys

##### General #####

package_path = join( dirname(dirname(realpath(__file__))), "src/biopython" )
sys.path.insert(0, package_path)

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'numpydoc']

templates_path = ['templates']
source_suffix = ['.rst']
master_doc = 'index'

project = 'Biopython 2.0'
copyright = '2017, Patrick Kunzmann'
author = 'Patrick Kunzmann'
version = '2.0'
release = '2.0a1'

exclude_patterns = ['build']

pygments_style = 'sphinx'

todo_include_todos = False


##### HTML #####

html_theme = 'alabaster'
html_static_path = ['static']
htmlhelp_basename = 'BiopythonDoc'


##### LaTeX #####

latex_elements = {}
latex_documents = [
    (master_doc, 'Biopython.tex', 'Biopython Documentation',
     'Patrick Kunzmann', 'manual'),
]