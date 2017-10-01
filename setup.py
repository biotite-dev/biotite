# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from setuptools import setup, find_packages, Extension
from setuptools.command.test import test as TestCommand
import sys
import shlex


release = "2.0a2"


if "sdist" in sys.argv:
    # Source distributions do not have extension modules
    # and therefore are using not C-accelerated functions
    ext_modules = None
else:
    from Cython.Build import cythonize
    try:
        ext_modules = cythonize(
            [Extension("biopython.sequence.align.calign",
                ["src/biopython/sequence/align/calign.pyx"]
             ),
             Extension("biopython.structure.io.pdbx.cprocessloop",
                ["src/biopython/structure/io/pdbx/cprocessloop.pyx"]
             ),
             Extension("biopython.cextensions",
                ["src/biopython/cextensions.pyx"]
             )]
        )
    except ValueError:
        # In case of installing a source distribution,
        # the *.pyx files cannot be found
        ext_modules = None


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


setup(name="biopython",
    version = release,
    description = "A set of general tools for computational biology",
    author = "The Biopython contributors",
    url = "https://github.com/padix-key/biopython2",
    
    zip_safe = False,
    packages = find_packages("src"),
    package_dir = {"" : "src"},
    
    ext_modules = ext_modules,
    
    # Including substitution matrix data
    package_data = {"biopython.sequence.align" : ["matrix_data/*.mat"]},
    
    install_requires = ["requests",
                        "numpy",
                        "scipy",
                        "matplotlib"],
    extras_require = {'trajectory':  ["mdtraj"],
    },
    
    cmdclass = {"test": PyTestCommand},
    tests_require = ["pytest"],
    
    command_options = {
        'build_sphinx':
            {"source_dir" : ("setup.py", "./doc"),
             "build_dir"  : ("setup.py", "./doc/_build"),
             "release"    : ("setup.py", "2.0a2")}
    }
)