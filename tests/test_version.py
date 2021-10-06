import re
from os.path import join, dirname, realpath
import pytest
from fastpdb import __version__
    

@pytest.mark.parametrize("file_path", ["pyproject.toml", "Cargo.toml"])
def test_version(file_path):
    abs_file_path = join(dirname(dirname(realpath(__file__))), file_path)
    with open(abs_file_path) as file:
        lines = file.read().splitlines()
    
    version = None
    for line in lines:
        if line.lstrip().startswith("version"):
            version_match = re.search('".*"', line)
            if version_match:
                # Remove quotes
                version = version_match.group(0)[1 : -1]
                break
    if version is None:
        raise ValueError(f"No version is specified in '{file_path}'")

    assert version == __version__
