# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, Tuple, Iterable, Iterator, TextIO, MutableMapping
from ....file import TextFile


class FastaFile(TextFile, MutableMapping[str, str]):
    def __init__(self, chars_per_line: int = 80) -> None: ...
