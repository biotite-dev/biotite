from setuptools import setup
from Cython.Build import cythonize
import numpy as np
import sys


def _should_build_wheel():
    for wheel_type in ["bdist_wheel", "editable_wheel"]:
        if wheel_type in sys.argv:
            return True
    return False


if _should_build_wheel():
    # Only build C files and compile them when building a wheel
    ext_modules = cythonize(
        "src/**/*.pyx",
        include_path=[np.get_include()],
    )
    for mod in ext_modules:
        mod.define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
        mod.include_dirs = [np.get_include()]

    setup(
        ext_modules=ext_modules,
    )
else:
    # In the source distribution, the `.pyx` files are considered the source files
    setup()
