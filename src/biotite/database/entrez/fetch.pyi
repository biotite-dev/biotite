# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import List, Union, overload


def get_database_name(database: str) -> str: ...

@overload
def fetch(
    uids: str,
    target_path: str,
    suffix: str,
    db_name: str,
    ret_type: str,
    ret_mode: str = "text",
    overwrite: bool = False,
    verbose: bool = False,
    mail: str = ""
) -> str: ...
@overload
def fetch(
    uids: List[str],
    target_path: str,
    suffix: str,
    db_name: str,
    ret_type: str,
    ret_mode: str = "text",
    overwrite: bool = False,
    verbose: bool = False,
    mail: str = ""
) -> List[str]: ...

def fetch_single_file(
    uids: List[str],
    file_name: str,
    db_name: str,
    ret_type: str,
    ret_mode: str = 'text',
    mail: None = None
) -> str: ...
