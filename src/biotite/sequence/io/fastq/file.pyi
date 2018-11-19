# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, Tuple, Iterable, Iterator, TextIO, MutableMapping
import numpy as np
from ....file import TextFile
from ..seqtypes import NucleotideSequence


class FastqFile(TextFile,
                MutableMapping[str, Union[NucleotideSequence, np.ndarray]]):
    def __init__(self, chars_per_line: int = 80) -> None: ...
    def get_sequence(self, identifier) -> NucleotideSequence: ...
    def get_quality(self, identifier) -> np.ndarray: ...