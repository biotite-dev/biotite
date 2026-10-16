from __future__ import annotations

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["SeedExtension"]

from typing import TYPE_CHECKING
from biotite.rust.sequence.align import SeedExtension

if TYPE_CHECKING:
    pass

# The class is implemented in Rust and therefore is not generic at runtime;
# allow ``SeedExtension[S1, S2]`` subscription so the annotated stub works
SeedExtension.__class_getitem__ = classmethod(lambda cls, params: cls)
