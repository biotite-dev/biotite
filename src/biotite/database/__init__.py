# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage for fetching data from online databases.

Each subpackage of the database package contains an interface for a
specific database.

One function, all subpackages share, is the `fetch()` function. It will
download one or multiple files that were specified via the function
arguments (e.g. PDB ID) into the specified directory. The return value
is the downloaded file name or a list of file names respectively.

If the file is only needed temporarily, they can be stored in a
temporary directory by using `biotite.temp_dir()` or
`biotite.temp_file()` as path name.
"""

__author__ = "Patrick Kunzmann"