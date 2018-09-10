"""
Big data - The Database subpackage
==================================

Biological databases are the backbone of computational biology. The
``database`` subpackage provides interfaces for popular online databases
like the RCSB PDB or the NCBI Entrez database.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

########################################################################
# Fetching structure files from the RCSB PDB
# ------------------------------------------
#
# Downloading structure files from the RCSB PDB is quite easy:
# Simply specify the PDB ID, the file format and the target directory
# for the `fetch()` function and you are done.
# The function even returns the path to the downloaded file, so you can
# just load it via the other *Biotite* subpackages (more on this later).
#
# For our later purposes, we will download on a protein structure as
# small as possible, namely the miniprotein *TC5b* (PDB: 1L2Y) into a
# temporary directory.

import biotite
import biotite.database.rcsb as rcsb
import biotite.structure.io as strucio
file_path = rcsb.fetch("1l2y", "pdb", biotite.temp_dir())