r"""
Creating an MMTF based PDB archive
==================================

In this example, all available PDB entries are downloaded in the MMTF
format and the files are added to an .tar archive.
Since the archive is a single file, tasks like copying or deleting are
much faster than for an actual directory.
For analysis, MMTF file objects can be extracted from the archive,
without actually extracting the MMTF file onto the hard drive.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import os
import os.path
import datetime
import concurrent.futures
import tarfile
import biotite.database.rcsb as rcsb
import biotite.structure.io.mmtf as mmtf


### Download of PDB and archive creation ###

# MMTF files are downloaded into a new directory in this path
# and the .tar archive is created here
base_path = "path/to/directoy"

# A Query class for getting all available PDB IDs 
class HoldingsQuery(rcsb.SimpleQuery):

    def __init__(self, method="ignore", molecule_type="ignore", has_data=None):
        super().__init__("HoldingsQuery")
        self.add_param("experimentalMethod", method)
        self.add_param("moleculeType", molecule_type)

# Obtain all PDB IDs
all_id_query = HoldingsQuery()
pdb_ids = rcsb.search(all_id_query)

# Name for donwload directory
now = datetime.datetime.now()
mmtf_dir = os.path.join(
    base_path, f"mmtf_{now.year:04d}{now.month:02d}{now.day:02d}"
)
if not os.path.isdir(mmtf_dir):
    os.mkdir(mmtf_dir)

# Download all PDB IDs with parallelized HTTP requests
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    for pdb_id in pdb_ids:
        executor.submit(rcsb.fetch, pdb_id, "mmtf", mmtf_dir)

# Create .tar archive file from MMTF files in directory
with tarfile.open(mmtf_dir+".tar", mode="w") as file:
    for pdb_id in pdb_ids:
        file.add(
            os.path.join(mmtf_dir, pdb_id+".mmtf"),
            pdb_id+".mmtf"
        )


### File access for analysis ###

# Iterate over all files in archive
# Instead of extracting the files from the archive,
# the .tar file is directly accessed
with tarfile.open(mmtf_dir+".tar", mode="r") as file:
    for member in file.getnames():
        mmtf_file = mmtf.MMTFFile()
        mmtf_file.read(file.extractfile(member))
        ###
        # Do some fancy stuff with the data...
        ###