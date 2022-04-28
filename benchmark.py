import time
import tempfile
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import biotite.database.rcsb as rcsb
import biotite.structure.info as info
import biotite.structure.io.pdb as pdb
import fastpdb


REPEATS = 1000
PDB_ID = "1AKI"
WIDTH = 0.25


# Call this function before the benchmark
# to avoid a bias due to the initial loading time
info.bond_dataset()


pdb_file_path = rcsb.fetch(PDB_ID, "pdb", tempfile.gettempdir())

fastpdb_runtimes = {}
biotite_runtimes = {}


now = time.time_ns()
for _ in range(REPEATS):
    pdb_file = fastpdb.PDBFile.read(pdb_file_path)
    pdb_file.get_coord(model=1)
fastpdb_runtimes["Read coord"] = (time.time_ns() - now) * 1e-6 / REPEATS

now = time.time_ns()
for _ in range(REPEATS):
    pdb_file = pdb.PDBFile.read(pdb_file_path)
    pdb_file.get_coord(model=1)
biotite_runtimes["Read coord"] = (time.time_ns() - now) * 1e-6 / REPEATS


now = time.time_ns()
for _ in range(REPEATS):
    pdb_file = fastpdb.PDBFile.read(pdb_file_path)
    pdb_file.get_structure(model=1)
fastpdb_runtimes["Read model"] = (time.time_ns() - now) * 1e-6 / REPEATS

now = time.time_ns()
for _ in range(REPEATS):
    pdb_file = pdb.PDBFile.read(pdb_file_path)
    pdb_file.get_structure(model=1)
biotite_runtimes["Read model"] = (time.time_ns() - now) * 1e-6 / REPEATS


pdb_file = pdb.PDBFile.read(pdb_file_path)
atoms = pdb_file.get_structure(model=1)

now = time.time_ns()
for _ in range(REPEATS):
    pdb_file = fastpdb.PDBFile()
    pdb_file.set_structure(atoms)
    pdb_file.write(tempfile.TemporaryFile("w"))
fastpdb_runtimes["Write model"] = (time.time_ns() - now) * 1e-6 / REPEATS

now = time.time_ns()
for _ in range(REPEATS):
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(atoms)
    pdb_file.write(tempfile.TemporaryFile("w"))
biotite_runtimes["Write model"] = (time.time_ns() - now) * 1e-6 / REPEATS


matplotlib.rc("font", size=12)
fig, ax = plt.subplots(figsize=(8.0, 4.0))

labels = list(fastpdb_runtimes.keys())
fastpdb_speedup = np.array(list(biotite_runtimes.values())) / \
                  np.array(list(fastpdb_runtimes.values()))

bars = ax.bar(
    np.arange(len(fastpdb_speedup)) - WIDTH/2, fastpdb_speedup,
    WIDTH, color="#0a6efd", linewidth=1.0, edgecolor="black", label="fastpdb"
)
ax.bar_label(bars, padding=3, fmt="%.1f×")
ax.bar(
    np.arange(len(fastpdb_speedup)) + WIDTH/2, np.ones(len(fastpdb_speedup)),
    WIDTH, color="#e1301d", linewidth=1.0, edgecolor="black", label="biotite"
)



ax.legend(loc="upper left", frameon=False)
ax.set_xticks(np.arange(len(fastpdb_runtimes)))
ax.set_xticklabels(labels)
ax.margins(y=0.1)
ax.set_ylabel("Speedup")
ax.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=1))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d×"))
fig.tight_layout()

plt.savefig("benchmark.svg")