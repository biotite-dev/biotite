import argparse
import json
import shutil
import textwrap
import zipfile
from io import BytesIO
from pathlib import Path
import requests

RELEASE_URL = "https://github.com/biotite-dev/biotite/releases/download/"

HTCACCESS = textwrap.dedent(
    r"""
    RewriteBase /
    RewriteEngine On
    # Redirect if page name does not start with 'latest' or version identifier
    RewriteRule ^(?!latest|\d+\.\d+\.\d+|robots.txt)(.*) latest/$1 [R=301,L]

    ErrorDocument 404 /latest/404.html
    """
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the complete content of the documentation website, including "
            "multiple versions of the documentation, and miscellaneous files."
        )
    )
    parser.add_argument(
        "input_json", help="The 'switcher.json' file to read the versions from."
    )
    parser.add_argument(
        "output_dir", help="The directory to write the documentation to."
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite the output directory."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        raise FileExistsError(f"Output directory '{output_dir}' already exists")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Download the documentation versions defined in the switcher JSON
    with open(args.input_json, "r") as file:
        switcher_json = json.load(file)
    latest_version = None
    for entry in switcher_json:
        version = entry["version"]
        if entry.get("preferred"):
            latest_version = version
        tag = "v" + version
        url = RELEASE_URL + tag + "/doc.zip"
        response = requests.get(url)
        with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall(output_dir)
        # The extracted directory is named 'doc' -> rename it to the version
        (output_dir / "doc").rename(output_dir / version)

    # Add 'latest'-link to the latest version
    if latest_version is None:
        raise ValueError("No preferred version specified in switcher.json")
    Path(output_dir / "latest").symlink_to(latest_version, target_is_directory=True)

    # Add the miscellaneous files
    with open(output_dir / ".htaccess", "w") as file:
        file.write(HTCACCESS)
    with open(output_dir / "robots.txt", "w") as file:
        file.write("User-agent: *\n")
        # Make search engines ignore the version-specific documentation URLs
        for entry in switcher_json:
            version = entry["version"]
            file.write(f"Disallow: /{version}/\n")
