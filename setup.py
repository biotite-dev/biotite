# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from setuptools import setup, find_packages, Extension
import sys


release = "2.0a2"


if "--no-extensions" in sys.argv:
    ext_modules = None
    sys.argv.remove("--no-extensions")
else:
    try:
        from Cython.Build import cythonize
        ext_modules = cythonize(
            [Extension("biopython.sequence.align.calign",
                       ["src/biopython/sequence/align/calign.pyx"]
                      ),
             Extension("biopython.structure.io.pdbx.cprocessloop",
                       ["src/biopython/structure/io/pdbx/cprocessloop.pyx"]
                      ),
             Extension("biopython.ccheckext",
                       ["src/biopython/ccheckext.pyx"]
                      )]
        )
    except:
        ext_modules = \
            [Extension("biopython.sequence.align.calign",
                       ["src/biopython/sequence/align/calign.c"]
                      ),
             Extension("biopython.structure.io.pdbx.cprocessloop",
                       ["src/biopython/structure/io/pdbx/cprocessloop.c"]
                      ),
             Extension("biopython.ccheckext",
                       ["src/biopython/ccheckext.c"]
                      )
            ]


setup(name="Biopython",
    version = release,
    description = "A set of general tools for computational biology",
    author = "The Biopython contributors",
    url = "https://github.com/padix-key/biopython2",
    
    zip_safe = False,
    packages = find_packages("src"),
    package_dir = {"" : "src"},
    
    ext_modules = ext_modules,
    
    # Including substitution matrix data
    package_data = {"" : ["*.npy"]},
    
    install_requires = ["requests",
                        "numpy",
                        "scipy",
                        "matplotlib"],
    extras_require = {'trajectory':  ["mdtraj"],
    },
    
    test_suite = "tests.main.test_suite",
    
    command_options = {
        'build_sphinx':
            {"source_dir" : ("setup.py", "./doc"),
             "build_dir"  : ("setup.py", "./doc/_build"),
             "release"    : ("setup.py", "2.0a2")}
    }
)