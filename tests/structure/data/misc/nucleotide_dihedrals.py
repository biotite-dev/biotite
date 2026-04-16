import json
from pathlib import Path
import barnaba

# `barnaba` does not properly handle discontinuities in the structure
# -> A structure without discontinuities is selected
STRUCTURE_FILE = Path(__file__).parents[1] / "4p5j.cif"
OUTPUT_FILE = Path(__file__).parent / "nucleotide_dihedrals.json"

# Order of angles in the array returned by `barnaba.backbone_angles()`
ANGLE_NAMES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]


if __name__ == "__main__":
    # `barnaba` returns angles only for canonical nucleotides, in the order they
    # appear in the structure. The single model of the structure is used.
    ba_angles, _ = barnaba.backbone_angles(STRUCTURE_FILE.as_posix())
    ba_angles = ba_angles[0]

    dihedral_dict = {
        name: ba_angles[:, i].tolist() for i, name in enumerate(ANGLE_NAMES)
    }
    with open(OUTPUT_FILE, "w") as file:
        json.dump(dihedral_dict, file, indent=4)
