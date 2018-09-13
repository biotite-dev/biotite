"""
Outsourcing - The Application subpackage
========================================

.. currentmodule:: biotite.application

Although you can achieve a lot with *Biotite*, there are still a lot of
things which are not implemented in this *Python* package.
But wait, this is what the :mod:`biotite.application` package is for:
It contains interfaces for popular external software.
This ranges from locally installed software to external tools running on
web servers.
The usage of these interfaces is seamless: Rather than writing input
files and reading output files, you simply put in your *Python*
objects (e.g. instances of :class:`Sequence` or :class:`AtomArray`) and
the interface returns *Python* objects (e.g. an :class:`Alignment`
object).

The base class for all interfaces is the :class:`Application` class.
Each :class:`Application` instance has a life cycle, starting with its
creation and ending with the result extraction.
Each state in this life cycle is described by the value of the 
*enum* :class:`AppState`, that each :class:`Application` contains:
Directly after its instantiation the app is in the ``CREATED`` state.
In this state further parameters can be set for the application run.
After the user calls the :func:`Application.start()` method, the app
state is set to ``RUNNING`` and the app performs the calculations.
When the application finishes the AppState
changes to ``FINISHED``.
The user can now call the :func:`Application.join()` method, concluding
the application in the ``JOINED`` state and making the results of the
application accessible.
Furthermore, this may trigger cleanup actions in some applications.
:func:`Application.join()` can even be called in the ``RUNNING`` state:
This will constantly check if the application has finished and will
directly go into the ``JOINED`` state as soon as the application reaches
the ``FINISHED`` state.
Calling the :func:`Application.cancel()` method while the application is
``RUNNING`` or ``FINISHED`` leaves the application in the ``CANCELLED``
state.
This triggers cleanup, too, but there are no accessible results.
If a method is called in an unsuitable app state, an
:class:`AppStateError` is called.
At each state in the life cycle, :class:`Application` type specific
methods are called, as shown in the following diagram.

.. image:: /static/assets/figures/app_lifecycle_path.svg

The following sample code shows how an :class:`Application` is generally
executed. Pay attention to the space between the
:func:`Application.start()` and :func:`Application.join()` method:
Since these are separate functions, you can do some other stuff, while
the :class:`Application` runs in the background.
Thus, an :class:`Application` behaves effectively like an additional
thread.
"""

from biotite.application import Application
# Create a dummy Application subclass
class MyApplication(Application):
    def __init__(self, param): super().__init__()
    def run(self): pass
    def is_finished(self): return True 
    def wait_interval(self): return 0.1
    def evaluate(self): pass
    def clean_up(self): pass
    def set_some_other_input(self, param): pass
    def get_some_data(self): return "some data"

param = "foo"
param2 = "bar"
app = MyApplication(param)
app.set_some_other_input(param2)
app.start()
# Time to do other stuff
app.join()
results = app.get_some_data()

########################################################################
# The following subsections will dive into the available
# :class:`Application` classes in depth.
# 
# Finding homologous sequences with BLAST
# ---------------------------------------
#
# .. currentmodule:: biotite.application.blast
#
# the :mod:`biotite.application.blast` subpackage provides an
# interface to NCBI BLAST: the :class:`BlastWebApp` class.
# Let's dive directly into the code, we try to find
# homologous sequences to the miniprotein *TC5b*:

import biotite.application.blast as blast
import biotite.sequence as seq
tc5b_seq = seq.ProteinSequence("NLYIQWLKDGGPSSGRPPPS")
app = blast.BlastWebApp("blastp", tc5b_seq)
app.start()
app.join()
alignments = app.get_alignments()
best_ali = alignments[0]
print(best_ali)
print()
print("HSP position in query: ", best_ali.query_interval)
print("HSP position in hit: ", best_ali.hit_interval)
print("Score: ", best_ali.score)
print("E-value: ", best_ali.e_value)
print("Hit UID: ", best_ali.hit_id)
print("Hit name: ", best_ali.hit_definition)

########################################################################
# This was too simple for BLAST:
# It just found the query sequence in the PDB.
# However, it gives a good impression about how this
# :class:`Application` works.
# Besides some optional parameters, the :class:`BlastWebApp` requires
# the BLAST program and the query sequence. After the app has finished,
# you get a list of alignments with descending score.
# An alignment is an instance of :class:`BlastAlignment`, a subclass of
# :class:`biotite.sequence.align.Alignment`.
# It contains some additional information as shown above.
# The hit UID can be used to obtain the complete hit sequence via
# :mod:`biotite.database.entrez`. 
# 
# The next alignment should be a bit more challenging.
# We take a random part of the *E. coli* BL21 genome and distort it a
# little bit.
# Since we still expect a high similarity to the original sequence,
# we decrease the E-value threshold.

import biotite.application.blast as blast
import biotite.sequence as seq
bl21_seq = seq.NucleotideSequence(
    "CGGAAGCGCTCGGTCTCCTGGCCTTATCAGCCACTGCGCGACGATATGCTCGTCCGTTTCGAAGA"
)
app = blast.BlastWebApp("blastn", bl21_seq, obey_rules=False)
app.set_max_expect_value(0.1)
app.start()
app.join()
alignments = app.get_alignments()
best_ali = alignments[0]
print(best_ali)
print()
print("HSP position in query: ", best_ali.query_interval)
print("HSP position in hit: ", best_ali.hit_interval)
print("Score: ", best_ali.score)
print("E-value: ", best_ali.e_value)
print("Hit UID: ", best_ali.hit_id)
print("Hit name: ", best_ali.hit_definition)

########################################################################
# In this snippet we have set :obj:`obey_rules` to false in the
# :class:`BlastWebApp` constructor, if we omitted this parameter and
# we started the last two code snippets in quick succession, a
# :class:`RuleViolationError` would be raised.
# This is because normally the :class:`BlastWebApp` respects NCBI's code of
# conduct and prevents you from submitting two queries within one
# minute. If you want to be rude to the NCBI server, create the
# instance with :obj:`obey_rules` set to false.
# 
# Multiple sequence alignments
# ----------------------------
#
# .. currentmodule:: biotite.application.muscle
#
# For *multiple sequence alignments* (MSAs) :mod:`biotite.application`
# offers several interfaces to MSA software.
# For our example we choose the software MUSCLE:
# The subpackage :mod:`biotite.application.muscle` contains the class
# :class:`MuscleApp` that does the job.
# You simply input the sequences you want to have aligned, run the
# application and get the resulting :class:`Alignment` object.

import biotite.application.muscle as muscle
import biotite.sequence as seq
seq1 = seq.ProteinSequence("BIQTITE")
seq2 = seq.ProteinSequence("TITANITE")
seq3 = seq.ProteinSequence("BISMITE")
seq4 = seq.ProteinSequence("IQLITE")
app = muscle.MuscleApp([seq1, seq2, seq3, seq4])
app.start()
app.join()
alignment = app.get_alignment()
print(alignment)

########################################################################
# For the lazy people there is also a convenience method,
# that handles the :class:`Application` execution internally.

alignment = muscle.MuscleApp.align([seq1, seq2, seq3, seq4])

########################################################################
# The alternatives to MUSCLE are Clustal-Omega and MAFFT.
# To use them, simply replace :class:`MuscleApp` with
# :class:`ClustalOmegaApp` or :class:`MafftApp` and you are done.

import biotite.application.clustalo as clustalo
alignment = clustalo.ClustalOmegaApp.align([seq1, seq2, seq3, seq4])
print(alignment)

########################################################################
# As shown in the output, the alignment with Clustal-Omega slightly
# differs from the one performed with MUSCLE.
# In contrast to MUSCLE, Clustal-Omega and MAFFT also support alignments
# of :class:`NucleotideSequence` instances.
# 
# Secondary structure annotation
# ------------------------------
# 
# .. currentmodule:: biotite.application.dssp
#
# Althogh :mod:`biotite.structure` offers the function
# :func:`annotate_sse()` to assign secondary structure elements based on
# the P-SEA algorithm, DSSP can also be used via the
# :mod:`biotite.application.dssp` subpackage (provided that DSSP is
# installed).
# Let us demonstrate this on the example of the good old miniprotein
# *TC5b*.
#
# Similar to the MSA examples, :class:`DsspApp` has the convenience
# function :func:`DsspApp.annotate_sse()`, which handles the
# :class:`DsspApp` execution internally.

import biotite
import biotite.database.rcsb as rcsb
import biotite.application.dssp as dssp
import biotite.structure.io as strucio
file_path = rcsb.fetch("1l2y", "mmtf", biotite.temp_dir())
stack = strucio.load_structure(file_path)
array = stack[0]
app = dssp.DsspApp(array)
app.start()
app.join()
sse = app.get_sse()
print(sse)