# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Dict, List, NewType, Union, Tuple
from ..alphabet import Alphabet


Color = Union[str, Tuple[float, float, float]]


def load_color_scheme(file_name: str) -> Dict: ... 

def get_color_scheme(
    name: str, alphabet: Alphabet, default: Color = ...
) -> List[Color]: ...

def list_color_scheme_names(alphabet: Alphabet) -> List[str]: ...