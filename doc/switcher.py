# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["create_switcher_json"]

import json
import requests
from packaging.version import Version
import biotite

RELEASE_REQUEST = "https://api.github.com/repos/biotite-dev/biotite/releases"
BIOTITE_URL = "https://www.biotite-python.org"


def _get_previous_versions(min_tag, n_versions, current_version):
    # The current version might already be released on GitHub
    # -> request one more version than necessary
    response = requests.get(RELEASE_REQUEST, params={"per_page": n_versions + 1})
    release_data = json.loads(response.text)
    versions = [Version(release["tag_name"]) for release in release_data]
    applicable_versions = [
        version
        for version in versions
        if version >= Version(min_tag) and version < current_version
    ]
    return applicable_versions[:n_versions]


def _get_current_version():
    return Version(biotite.__version__)


def create_switcher_json(file_path, min_tag, n_versions):
    """
    Create the version switcher JSON file for the documentation.

    Parameters
    ----------
    file_path : str or Path
        The path to the JSON file.
    min_tag : str
        The minimum version tag to be included.
    n_versions : int
        The maximum number of previously released versions to be included.
    """
    version_config = []
    current_version = _get_current_version()
    versions = _get_previous_versions(min_tag, n_versions, current_version)
    if current_version not in versions:
        versions.append(current_version)
    versions.sort()
    for version in versions:
        if version.micro != 0:
            # Documentation is not uploaded for patch versions
            continue
        version_config.append(
            {
                "name": f"{version.major}.{version.minor}",
                "version": str(version),
                "url": f"{BIOTITE_URL}/{version}/",
            }
        )
    # Mark the latest version as preferred
    if len(version_config) > 0:
        version_config[-1]["preferred"] = True
    with open(file_path, "w") as file:
        json.dump(version_config, file, indent=4)
