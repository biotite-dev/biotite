# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import os.path

files = ["prot.fasta", "nuc.fasta", "invalid.fasta"]

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

data_files = [os.path.join(data_dir, file) for file in files] 



