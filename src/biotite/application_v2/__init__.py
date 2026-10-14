"""
DEPRECATED: This subpackage has been renamed to :mod:`biotite.application`.
It merely forwards to :mod:`biotite.application` and will be removed in a
future release.
"""

__name__ = "biotite.application_v2"
__author__ = "Patrick Kunzmann"

import importlib
import sys
import warnings
from biotite.application import *  # noqa: F403

warnings.warn(
    "'biotite.application_v2' is deprecated, use 'biotite.application' instead",
    DeprecationWarning,
    stacklevel=2,
)

# Alias the subpackages, so that e.g. 'biotite.application_v2.dssp' resolves to
# the very same module object as 'biotite.application.dssp'
_SUBPACKAGES = [
    "autodock",
    "clustalo",
    "dssp",
    "mafft",
    "mmseqs",
    "muscle",
    "sra",
    "tantan",
    "viennarna",
]
for _subpackage in _SUBPACKAGES:
    _module = importlib.import_module(f"biotite.application.{_subpackage}")
    sys.modules[f"{__name__}.{_subpackage}"] = _module
    globals()[_subpackage] = _module
