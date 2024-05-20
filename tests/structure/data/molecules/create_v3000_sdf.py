from pathlib import Path
from rdkit import Chem

SCRIPT_PATH = Path(__file__).parent

for sdf_path in SCRIPT_PATH.glob("*.sdf"):
    if "v3000" in str(sdf_path):
        continue
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    writer = Chem.SDWriter(sdf_path.with_suffix(".v3000.sdf"))
    writer.SetForceV3000(True)
    for molecule in supplier:
        writer.write(molecule)
    writer.close()