# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

release = "2.0a2"

setup(name="Biopython",
    version = release,
    description = "A set of general tools for computational biology",
    author = "The Biopython contributors",
    url = "https://github.com/padix-key/biopython2",
    packages = find_packages("src"),
    package_dir = {"" : "src"},
    ext_modules = cythonize(
        [Extension  ("biopython.sequence.align.calign",
                        ["src/biopython/sequence/align/calign.pyx"])]
    ),
    package_data = {"biopython.sequence.align" : ["matrix_data/*.npy"]},
    install_requires = ["requests",
                        "numpy",
                        "scipy",
                        "matplotlib"],
    test_suite = "tests.main.test_suite",
    zip_safe = False,
    
    command_options = {
        'build_sphinx':
            {"source_dir" : ("setup.py", "./doc"),
             "build_dir"  : ("setup.py", "./doc/_build"),
             "release"    : ("setup.py", "2.0a2")}
    }
)