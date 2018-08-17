# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import List, Union, overload


@overload
def fetch(
    pdb_ids: str,
    format: str,
    target_path: str,
    overwrite: bool = False,
    verbose: bool = False
) -> str: ...
@overload
def fetch(
    pdb_ids: List[str],
    format: str,
    target_path: str,
    overwrite: bool = False,
    verbose: bool = False
) -> List[str]: ...