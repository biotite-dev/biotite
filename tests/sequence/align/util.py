# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import io
import json
import tarfile
from tests.util import data_dir


def legacy_alignments():
    """
    Iterate over the ``(param, alignment)`` pairs of the legacy consistency data
    set (``tests/sequence/data/legacy_consistency``) for all alignment methods.

    `param` is the ``params.json`` entry (including the ``method``), augmented
    with the ``file_name`` of the alignment within ``legacy_alignments.tar.gz``.
    `alignment` is the reference alignment loaded from that archive member.
    The pairs are sorted by ``file_name``.
    """
    # Import in function to avoid 'ModuleNotFoundError',
    # if a Cython module is not compiled yet
    import biotite.sequence.io.fasta as fasta

    base_dir = data_dir("sequence") / "legacy_consistency"
    with open(base_dir / "params.json") as file:
        params = json.load(file)

    pairs = []
    with tarfile.open(base_dir / "legacy_alignments.tar.gz", "r:gz") as tar:
        for file_name in sorted(params):
            entry = params[file_name]
            with io.TextIOWrapper(
                tar.extractfile(file_name), encoding="utf-8"
            ) as fasta_io:
                alignment = fasta.get_alignment(fasta.FastaFile.read(fasta_io))
            pairs.append(({"file_name": file_name, **entry}, alignment))
    return pairs
