import os
import shutil
import sys
import numpy as np
from Cython.Build import cythonize
from puccinialin import setup_rust
from setuptools import setup
from setuptools_rust import RustExtension


def _should_build_wheel():
    for wheel_type in ["bdist_wheel", "editable_wheel"]:
        if wheel_type in sys.argv:
            return True
    return False


if _should_build_wheel():
    if not os.environ.get("BIOTITE_OMIT_RUST", False):
        if not shutil.which("cargo"):
            # Rust compiler is not installed -> Install it temporarily
            extra_env = setup_rust()
            env = {**os.environ, **extra_env}
        else:
            env = None
        rust_extensions = [
            RustExtension(
                target="biotite.rust",
                path="Cargo.toml",
                env=env,
            )
        ]
    else:
        rust_extensions = None

    if not os.environ.get("BIOTITE_OMIT_CYTHON", False):
        # Only build C files and compile them when building a wheel
        cython_extensions = cythonize(
            "src/**/*.pyx",
            include_path=[np.get_include()],
        )
        for mod in cython_extensions:
            mod.define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
            mod.include_dirs = [np.get_include()]
    else:
        cython_extensions = None

    setup(
        ext_modules=cython_extensions,
        rust_extensions=rust_extensions,
    )
else:
    # In the source distribution, the `.pyx` files are considered the source files
    setup()
