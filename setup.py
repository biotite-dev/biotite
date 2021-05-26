# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import re
from os.path import join, abspath, dirname, normpath
import fnmatch
import os
from setuptools import setup, find_packages, Extension
import numpy
from Cython.Build import cythonize

original_wd = os.getcwd()
# Change directory to setup directory to ensure correct file identification
os.chdir(dirname(abspath(__file__)))

# Simply import long description from README file
with open("README.rst") as readme:
    long_description = readme.read()

# Parse the top level package for the version
# Do not use an import to prevent side effects
# e.g. required runtime dependencies
with open(join("src", "biotite", "__init__.py")) as init_file:
    for line in init_file.read().splitlines():
        if line.lstrip().startswith("__version__"):
            version_match = re.search('".*"', line)
            if version_match:
                # Remove quotes
                version = version_match.group(0)[1 : -1]
            else:
                raise ValueError("No version is specified in '__init__.py'")

# Compile Cython into C
try:
    cythonize(
        "src/**/*.pyx",
        include_path=[numpy.get_include()],
        language_level=3
    )
except ValueError:
    # This is a source distribution and the directory already contains
    # only C files
    pass


def get_extensions():
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
    return ext_modules


setup(
    name="biotite",
    version = version,
    description = ("A comprehensive library for "
                   "computational molecular biology"),
    long_description = long_description,
    author = "The Biotite contributors",
    license = "BSD 3-Clause",
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    url = "https://www.biotite-python.org",
    project_urls = {
        "Documentation": "https://biotite.biotite-python.org",
        "Repository": "https://github.com/biotite-dev/biotite",
    },
    
    zip_safe = False,
    packages = find_packages("src"),
    package_dir = {"" : "src"},
    
    ext_modules = get_extensions(),
    
    # Including additional data
    package_data = {
        # Substitution matrices
        "biotite.sequence.align"    : ["matrix_data/*.mat"],
        # Color schmemes
        "biotite.sequence.graphics" : ["color_schemes/*.json"],
        # Codon tables
        "biotite.sequence"          : ["codon_tables.txt"],
        # Structure data (masses, bonds, etc.)
        "biotite.structure.info"    : ["*.json", "*.msgpack"]
    },
    
    install_requires = ["requests >= 2.12",
                        "numpy >= 1.19",
                        "msgpack >= 0.5.6",
                        "networkx >= 2.0"],
    python_requires = ">=3.6",
    
    tests_require = ["pytest"],
)


# Return to original directory
os.chdir(original_wd)

