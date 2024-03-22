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

# Parse the top level package for the version
# Do not use an import to prevent side effects
# e.g. required runtime dependencies
version = None
with open(join("src", "biotite", "__init__.py")) as init_file:
    for line in init_file.read().splitlines():
        if line.lstrip().startswith("__version__"):
            version_match = re.search('".*"', line)
            if version_match:
                # Remove quotes
                version = version_match.group(0)[1 : -1]
            else:
                raise ValueError("No version is specified in '__init__.py'")
if version is None:
    raise ValueError("Unable to identify 'version' in __init__.py")

# Compile Cython into C
try:
    cythonize(
        "src/**/*.pyx",
        include_path=[numpy.get_include()],
        language_level=3
    )
except ValueError:
    # This is a source distribution and the directory already contains
    # only C/C++ files
    pass


def get_extensions():
    ext_sources = []
    for dirpath, _, filenames in os.walk(normpath("src/biotite")):
        for filename in (
            fnmatch.filter(filenames, "*.c") +
            fnmatch.filter(filenames, "*.cpp")
        ):
            ext_sources.append(os.path.join(dirpath, filename))
    ext_names = [source
                 .replace("src"+normpath("/"), "")
                 .replace(".cpp", "")
                 .replace(".c", "")
                 .replace(normpath("/"), ".")
                 for source in ext_sources]
    ext_modules = [Extension(ext_names[i], [ext_sources[i]],
                             include_dirs=[numpy.get_include()])
                   for i in range(len(ext_sources))]
    return ext_modules


setup(
    version = version,
    zip_safe = False,
    packages = find_packages("src"),
    package_dir = {"" : "src"},

    ext_modules = get_extensions(),

    # Including additional data
    package_data = {
        "biotite.sequence.align"    : [
            # Substitution matrices
            "matrix_data/*.mat",
            # Prime hash table sizes
            "primes.txt"
        ],
        # Color schmemes
        "biotite.sequence.graphics" : ["color_schemes/*.json"],
        # Codon tables
        "biotite.sequence"          : ["codon_tables.txt"],
        # Structure data (masses, bonds, etc.)
        "biotite.structure.info"    : ["atom_masses.json", "ccd/*"]
    },
)


# Return to original directory
os.chdir(original_wd)
