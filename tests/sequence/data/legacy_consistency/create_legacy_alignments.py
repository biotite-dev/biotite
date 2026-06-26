"""
Create legacy alignment files from the Cython implementations.

This script must be run while the Cython implementations of
``align_optimal()``, ``align_banded()`` and ``align_local_gapped()`` are still
active.  It will refuse to run if any of these functions has already been
replaced by a Rust implementation.

All alignments are written as FASTA members into a single
``legacy_alignments.tar.gz`` archive.  ``params.json`` maps each member file name
to its ``method`` (the alignment function), the ``seq_indices`` of the sequence
pair and the ``params`` that can be passed to the function as keyword arguments.
"""

import io
import itertools
import json
import tarfile
from pathlib import Path
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.io.fasta as fasta

BAND_WIDTH = 100
THRESHOLD = 100
# The gap penalty is part of the Cartesian product: affine and linear.
GAP_PENALTIES = [("affine", (-10, -1)), ("linear", -10)]
# The affine penalty is used to derive the band and seed for the dependent
# `align_banded()` and `align_local_gapped()` reference alignments.
REFERENCE_GAP_PENALTY = (-10, -1)

DATA_DIR = Path(__file__).resolve().parent
CAS9_PATH = DATA_DIR.parent / "cas9.fasta"
ALIGNMENT_PATH = DATA_DIR / "legacy_alignments.tar.gz"
PARAMS_PATH = DATA_DIR / "params.json"


def _assert_cython(func):
    """
    Raise if `func` is not a Cython function.
    """
    type_name = type(func).__name__
    if type_name != "cython_function_or_method":
        raise RuntimeError(
            "This script must be run with the legacy Cython implementation."
        )


def _middle_aligned_position(alignment):
    """
    Return the ``(seq1_pos, seq2_pos)`` tuple from the middle of the aligned
    (non-gap) positions in the alignment.
    """
    trace = alignment.trace
    aligned_mask = (trace != -1).all(axis=1)
    aligned_positions = trace[aligned_mask]
    mid = aligned_positions[len(aligned_positions) // 2]
    return mid[0].item(), mid[1].item()


def _add_alignment(tar, alignment, seq_names, file_name):
    """
    Write `alignment` as a FASTA member named `file_name` into the open tar
    archive `tar`.
    """
    fasta_file = fasta.FastaFile()
    fasta.set_alignment(fasta_file, alignment, seq_names)
    text = io.StringIO()
    fasta_file.write(text)
    data = text.getvalue().encode()
    info = tarfile.TarInfo(file_name)
    info.size = len(data)
    # Use a fixed timestamp to keep the archive reproducible
    info.mtime = 0
    tar.addfile(info, io.BytesIO(data))


def main():
    _assert_cython(align.align_optimal)
    _assert_cython(align.align_banded)
    _assert_cython(align.align_local_gapped)

    fasta_file = fasta.FastaFile.read(CAS9_PATH)
    headers = list(fasta_file.keys())
    sequences = [seq.ProteinSequence(sequence) for sequence in fasta_file.values()]
    matrix = align.SubstitutionMatrix.std_protein_matrix()

    params = {}

    with tarfile.open(ALIGNMENT_PATH, "w:gz") as tar:
        for i, j in itertools.combinations(range(len(sequences)), 2):
            seq1, seq2 = sequences[i], sequences[j]
            seq_names = [headers[i], headers[j]]

            # A global optimal alignment is used to derive the band (for
            # `align_banded`) and the seed (for `align_local_gapped`).
            reference_alignment = align.align_optimal(
                seq1, seq2, matrix, gap_penalty=REFERENCE_GAP_PENALTY, max_number=1
            )[0]
            mid_pos1, mid_pos2 = _middle_aligned_position(reference_alignment)
            diag = mid_pos2 - mid_pos1
            band = (diag - BAND_WIDTH, diag + BAND_WIDTH)
            seed = (mid_pos1, mid_pos2)

            # --- align_optimal: local x terminal_penalty x gap_penalty ---
            for local, terminal_penalty, (gap_name, gap_penalty) in itertools.product(
                (False, True), (False, True), GAP_PENALTIES
            ):
                alignment = align.align_optimal(
                    seq1,
                    seq2,
                    matrix,
                    gap_penalty=gap_penalty,
                    terminal_penalty=terminal_penalty,
                    local=local,
                    max_number=1,
                )[0]
                name = "local" if local else "global"
                stem = f"align_optimal_{i}_{j}_{name}_{gap_name}"
                if terminal_penalty:
                    stem += "_terminal"
                file_name = f"{stem}.fasta"
                _add_alignment(tar, alignment, seq_names, file_name)
                params[file_name] = {
                    "method": "align_optimal",
                    "seq_indices": [i, j],
                    "params": {
                        "gap_penalty": gap_penalty,
                        "local": local,
                        "terminal_penalty": terminal_penalty,
                    },
                }

            # --- align_banded: local x gap_penalty ---
            for local, (gap_name, gap_penalty) in itertools.product(
                (False, True), GAP_PENALTIES
            ):
                alignment = align.align_banded(
                    seq1,
                    seq2,
                    matrix,
                    band=band,
                    gap_penalty=gap_penalty,
                    local=local,
                    max_number=1,
                )[0]
                name = "local" if local else "global"
                file_name = f"align_banded_{i}_{j}_{name}_{gap_name}.fasta"
                _add_alignment(tar, alignment, seq_names, file_name)
                params[file_name] = {
                    "method": "align_banded",
                    "seq_indices": [i, j],
                    "params": {
                        "gap_penalty": gap_penalty,
                        "band": list(band),
                        "local": local,
                    },
                }

            # --- align_local_gapped: gap_penalty (always local) ---
            for gap_name, gap_penalty in GAP_PENALTIES:
                alignment = align.align_local_gapped(
                    seq1,
                    seq2,
                    matrix,
                    seed=seed,
                    threshold=THRESHOLD,
                    gap_penalty=gap_penalty,
                    max_number=1,
                )[0]
                file_name = f"align_local_{i}_{j}_{gap_name}.fasta"
                _add_alignment(tar, alignment, seq_names, file_name)
                params[file_name] = {
                    "method": "align_local_gapped",
                    "seq_indices": [i, j],
                    "params": {
                        "gap_penalty": gap_penalty,
                        "seed": list(seed),
                        "threshold": THRESHOLD,
                    },
                }

    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()
