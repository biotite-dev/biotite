Outsourcing - The Application subpackage
----------------------------------------

Although you can achieve a lot with *Biotite*, there are still a lot of
things which are not implemented in this *Python* package. But wait, this is
what the ``application`` package is for: It contains interfaces for popular
external software. This ranges from locally installed software to tools
running on servers. The usage of these interfaces is seamless: Rather than
writing input files and reading output files, you simply put in your Python
objects (e.g. instances of `Sequence` or `AtomArray`) and the interface
returns *Python* objects (e.g. an `Alignment` object).

The base class for all interfaces is the `Application` class. Each
`Application` instance has a life cycle, starting with its creation and ending
with the result extraction. Each state in this life cycle is described by
the instance of the `enum` `AppState`, that each `Application` contains.
Directly after its instantiation the app is in the *CREATED* state. In this
state further parameters can be set for the application run. After the user
calls the `start()` method, the app state is set to *RUNNING* and the
app performs the calculations. When the application finishes the AppState
changes to *FINISHED*. The user can now call the `join()` method,
concluding the application in the *JOINED* state and making the results of the
application accessible. Furthermore, this may trigger cleanup actions in some
applications. `join()` can even be called in the *RUNNING* state: This will
constantly check if the application has finished and will directly go into
the *JOINED* state as soon as the application reaches the *FINISHED* state.
Calling the `cancel()` method while the application is *RUNNING* or *FINISHED*
leaves the application in the *CANCELLED* state. This triggers cleanup, too,
but there are no accessible results. If a method is called in an unsuitable app
state, an `AppStateError` is called. At each state in the life cycle,
`Application` type specific methods are called, as shown in the following
diagram.

.. image:: /static/assets/figures/app_lifecycle_path.svg

The following sample code shows how an `Application` is generally executed.
Pay attention to the space between the `run()` and `join()` method: Since these
are separate functions, you can do some other stuff, while the `Application`
runs in the background. Therefore, an `Application` behaves effectively like an
additional thread.

.. code-block:: python

   app = MyApplication(param1, param2)
   app.set_some_other_input(param)
   app.start()
   # Time to do other stuff
   app.join()
   results = app.get_some_data()

The following section will dive into the available `Application` classes in
depth.

Finding homologous sequences with BLAST
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

the ``application.blast`` subpackage provides an interface to NCBI BLAST: the
`BlastWebApp` class. Let's dive directly into the code, we try to find
homologous sequences to the miniprotein *TC5b*:

.. code-block:: python
   
   import src.biotite.application.blast as blast
   import src.biotite.sequence as seq
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

Output:

.. code-block:: none
   
   NLYIQWLKDGGPSSGRPPPS
   NLYIQWLKDGGPSSGRPPPS
   
   HSP position in query:  (1, 20)
   HSP position in hit:  (1, 20)
   Score:  101
   E-value:  2.9064e-05
   Hit UID:  1L2Y_A
   Hit name:  Chain A, Nmr Structure Of Trp-Cage Miniprotein Construct Tc5b

This was too simple for BLAST: It just found the query sequence in the PDB.
However, it gives a good impression about how this `Application` works.
Besides some optional parameters, the `BlastWebApp` requires the BLAST
program and the query sequence. After the app has finished, you get
a list of alignments with descending score. An alignment is an instance of
`BlastAlignment`, a subclass of `Alignment` in ``sequence.align``. It
contains some additional information as shown above. The hit UID can be used
to obtain the complete hit sequence via ``database.entrez``. 

The next alignment should be a bit more challenging. We take a random part of
the *E. coli* BL21 genome and distort it a little bit. Since we still expect a
high similarity to the original sequence, we decrease the E-value threshold.

.. code-block:: python
   
   import src.biotite.application.blast as blast
   import src.biotite.sequence as seq
   bl21_seq = seq.NucleotideSequence(
       "CGGAAGCGCTCGGTCTCCTGGCCTTATCAGCCACTGCGCGACGATATGCTCGTCCGTTTCGAAGA"
   )
   app = blast.BlastWebApp("blastn", bl21_seq)
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

Output:

.. code-block:: none
   
   CGGAAGCGCTCGGTCTCCTGGCC----TTATCAGCCACTGCGCGACGATATGCTCGTCCGTTTCGAAGA
   CGGAAGCGCT-GGTC-CCTGCCCCGCTTTATCAGGGAATGCGCGACGGCAAAATCGTCCGTTTCGAAGA
   
   HSP position in query:  (1, 65)
   HSP position in hit:  (2915867, 2915933)
   Score:  54
   E-value:  0.0044495
   Hit UID:  CP023383
   Hit name:  Escherichia coli strain 1223 chromosome, complete genome

If we started the last two code snippets in quick succession, a
`RuleViolationError` would be raised. This is because the `Application`
respects NCBI's code of conduct and prevents you from submitting two queries
within one minute. If you want to be rude to the NCBI server, create the
instance with ``obey_rules=False``.

Multiple sequence alignments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For *multiple sequence alignments* (MSAs) the ``application.muscle`` subpackage
offers the `MuscleApp` for the MSA software MUSCLE. You simply input the
sequences you want to have aligned, run the application and get the resulting
`Alignment` object (you already know from ``sequence.align``):

.. code-block:: python

   import src.biotite.application.muscle as muscle
   import src.biotite.sequence as seq
   seq1 = seq.ProteinSequence("BIQTITE")
   seq2 = seq.ProteinSequence("TITANITE")
   seq3 = seq.ProteinSequence("BISMITE")
   seq4 = seq.ProteinSequence("IQLITE")
   app = muscle.MuscleApp([seq1, seq2, seq3, seq4], bin_path="muscle")
   app.start()
   app.join()
   alignment = app.get_alignment()
   print(alignment)

Output:

.. code-block:: none

   BIQT-ITE
   TITANITE
   BISM-ITE
   -IQL-ITE

As of now, this does only work with protein sequences.


