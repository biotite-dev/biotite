from typing import List


def fetch(
    uids: str,
    target_path: str,
    suffix: str,
    db_name: str,
    ret_type: str,
    ret_mode: str = 'text',
    overwrite: bool = False,
    verbose: bool = False,
    mail: str = ''
) -> str: ...


def fetch_single_file(
    uids: List[str],
    file_name: str,
    db_name: str,
    ret_type: str,
    ret_mode: str = 'text',
    mail: None = None
) -> str: ...
