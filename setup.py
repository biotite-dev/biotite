# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

from setuptools import setup, find_packages, Extension
from setuptools.command.test import test as TestCommand
import sys
import shlex
import glob
from os.path import join, abspath, dirname, normpath
import fnmatch
import os
from src.biotite import __version__

long_description = """
The Biotite package bundles popular tools in computational biology into an
unifying framework. It offers file I/O operations, analyses and manipulations
for biological sequence and structure data. Furthermore, the package provides
interfaces for popular biological databases and external software.

The internal structure and sequence representations are based on *NumPy*
`ndarrays`, taking the advantage of C-accelerated operations. Time consuming
operations that could not be vectorised are mostly implemented in *Cython* in
order to achieve C-accelerations in those places, too.

Additionally the package aims for simple usability and extensibility: The
objects representing structures and sequences can be indexed and scliced like
an `ndarray`. Even the actual internal `ndarrays` are easily accessible
allowing advanced users to implement their own algorithms upon the existing
types.
"""

import numpy
try:
    from Cython.Build import cythonize
    cythonize("src/**/*.pyx", include_path=[numpy.get_include()])
except ValueError:
    pass

def get_extensions():
    original_wd = os.getcwd()
    # Change directory to setup directory to ensure correct globbing
    os.chdir(dirname(abspath(__file__)))
    ext_sources = []
    for dirpath, dirnames, filenames in os.walk(normpath("src/biotite")):
        for filename in fnmatch.filter(filenames, '*.c'):
            ext_sources.append(os.path.join(dirpath, filename))
    ext_names = [source
                 .replace("src"+normpath("/"), "")
                 .replace(".c", "")
                 .replace(normpath("/"), ".")
                 for source in ext_sources]
    ext_modules = [Extension(ext_names[i], [ext_sources[i]],
                             include_dirs=[numpy.get_include()])
                   for i in range(len(ext_sources))]
    # Return to original directory
    os.chdir(original_wd)
    return ext_modules


class PyTestCommand(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        super().initialize_options()
        self.pytest_args = ''

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


setup(
    name="biotite",
    version = __version__,
    description = "A general framework for computational biology",
    long_description = long_description,
    author = "The Biotite contributors",
    url = "https://github.com/biotite-dev/biotite",
    license = "BSD 3-Clause",
    classifiers = (
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ),
    
    zip_safe = False,
    packages = find_packages("src"),
    package_dir = {"" : "src"},
    
    ext_modules = get_extensions(),
    
    # Including substitution matrix data
    package_data = {"biotite.sequence.align" : ["matrix_data/*.mat"],
                    "biotite.sequence" : ["codon_tables.txt"],},
    
    install_requires = ["requests",
                        "numpy",
                        "matplotlib",
                        "msgpack"],
    python_requires = ">=3.4",
    
    cmdclass = {"test": PyTestCommand},
    tests_require = ["pytest"],
    
    command_options = {
        'build_sphinx':
            {"source_dir" : ("setup.py", "./doc"),
             "build_dir"  : ("setup.py", "./doc/_build"),
             "release"    : ("setup.py", __version__)}
    }
)
