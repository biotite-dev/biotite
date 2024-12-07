# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["create_switcher_json"]

import functools
import json
import re
from dataclasses import dataclass
import requests
import biotite

RELEASE_REQUEST = "https://api.github.com/repos/biotite-dev/biotite/releases"
BIOTITE_URL = "https://www.biotite-python.org"
SEMVER_TAG_REGEX = r"^v?(\d+)\.(\d+)\.(\d+)"


@functools.total_ordering
@dataclass(frozen=True)
class Version:
    major: ...
    minor: ...
    patch: ...

    @staticmethod
    def from_tag(tag):
        match = re.match(SEMVER_TAG_REGEX, tag)
        if match is None:
            raise ValueError(f"Invalid tag: {tag}")
        major, minor, patch = map(int, match.groups())
        return Version(major, minor, patch)

    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"

    def __ge__(self, other):
        return (self.major, self.minor, self.patch) >= (
            other.major,
            other.minor,
            other.patch,
        )


def _get_previous_versions(min_tag, n_versions, current_version):
    # The current version might already be released on GitHub
    # -> request one more version than necessary
    response = requests.get(RELEASE_REQUEST, params={"per_page": n_versions + 1})
    release_data = json.loads(response.text)
    versions = [Version.from_tag(release["tag_name"]) for release in release_data]
    applicable_versions = [
        version
        for version in versions
        if version >= Version.from_tag(min_tag) and version < current_version
    ]
    return applicable_versions[:n_versions]


def _get_current_version():
    return Version(*biotite.__version_tuple__[:3])


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
        if version.patch != 0:
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
    version_config[-1]["preferred"] = True
    with open(file_path, "w") as file:
        json.dump(version_config, file, indent=4)
