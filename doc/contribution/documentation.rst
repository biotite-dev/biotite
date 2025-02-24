Writing the documentation
=========================

Having a good documentation on a package is arguably as important as the
code itself.
Hence, *Biotite* provides in addition to the API reference documented in
docstrings, a comprehensive documentation residing in the ``doc/`` directory
containing tutorials, examples and more.

Documentation generation
------------------------
*Biotite* uses the widely used `Sphinx <https://www.sphinx-doc.org>`_ package
for generating its documentation.
Therefore, the documentation is based on *reStructuredText* files.
The line length of these ``*.rst`` files is also limited to 79 characters
where reasonable.

To build the documentation, run from the root directory of the repository:

.. code-block:: console

    $ sphinx-build doc build/doc

Documentation structure
-----------------------
*Biotite* employs the
`Divio documentation system <https://documentation.divio.com>`_.
In short, the documentation is splitted into four different parts that
addresses different purposes and audiences:

.. list-table:: Documentation sections
    :widths: 10 20 20
    :header-rows: 1

    * - Part
      - Summary
      - Section in *Biotite*
    * - `Tutorials <https://documentation.divio.com/tutorials.html>`_
      - Learning of basic concepts via simple examples
      - Tutorial
    * - `How-to guides <https://documentation.divio.com/how-to.html>`_
      - Step-by-step instructions for specific real-world tasks
      - Example gallery
    * - `Explanation <https://documentation.divio.com/explanation.html>`_
      - Detailed explanation of concepts
      - Literature citations, contributor guide
    * - `Reference <https://documentation.divio.com/reference.html>`_
      - Technical description in a consistent format
      - API reference

When adding new content, please consider which part of the documentation
it fits best and adhere to the purpose of that part.
You might also consider to split the content into multiple parts
(e.g. into an example for the gallery and a tutorial), if you think your
content fulfills a mixture of different purposes.

.. _example_gallery:

Example gallery
---------------
For gallery generation the package *sphinx-gallery* is used.
Please refer to its
`documentation <http://sphinx-gallery.readthedocs.io>`_
for further information on script formatting.
The example scripts are placed in ``doc/examples/scripts`` in the subdirectory
that fits best topic of the example.
Choose a title for the example that focuses on the employed method rather than
the biological context.
For example,
'*Homology search and multiple sequence alignment of protein sequences*'
would be a better name than
'*Similarities of lysozyme variants*'.

Building the example gallery for the first time may take a while, as all
scripts are executed.
To build the documentation without the gallery and the tutorial, run

.. code-block:: console

    $ sphinx-build -D plot_gallery=0 doc build/doc

You may also ask the *Biotite* maintainers to run the example script and check
the generated page, if building the gallery on your device is not possible.

Static images
^^^^^^^^^^^^^
Static images can be included by adding the following comment in the
corresponding code block:

.. code-block:: python

    # sphinx_gallery_static_image = <name_of_the_image>.png

The image file must be stored in the same directory as the example script.

Tutorial
--------
When adding new content for a broad audience, it is appreciated to update the
tutorial pages (``doc/tutorial/``) as well.
The tutorial uses `jupyter-sphinx <https://jupyter-sphinx.readthedocs.io>`_ to
run the code snippets and show the results.
This has the advantage that the output of code snippets is not static but
dynamically generated based on the current state of the *Biotite* source
code.

Make sure to add

.. code-block:: rst

    .. include:: /tutorial/preamble.rst

at the beginning of the tutorial page.

API reference
-------------
Each  *Biotite* subpackage has a dedicated reference page, describing
its classes and functions.
The categories and classes/functions that are assigned to it can be set
in ``doc/apidoc.json``.
Classes/functions that are not assigned to any category are placed in
the 'Miscellaneous' category or, if no class/function is assigned,
in the 'Content' category.

Citing articles
---------------
*Biotite* uses
`sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io>`_ for
creating references in docstrings, examples, etc.
The references are stored in ``doc/references.bib`` with citation keys
in ``[Author][year]`` format.
References are cited with the ``:footcite:`` role and the bibliography
is rendered where the ``.. footbibliography::`` directive is placed.

Adding articles to bibliography
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The recommended way to add articles to the bibliography is not to add them
directly to ``references.bib``, but to update the *Biotite*
`Zotero <https://www.zotero.org/>`_ library.
As this step is a bit more involved, you may also ask the *Biotite* maintainers
to add the article for you.

After installation of *Zotero* and
`Better BibTeX <https://retorque.re/zotero-better-bibtex/>`_, import the
`Biotite library <https://www.zotero.org/groups/5533833/biotite_documentation>`_.
Then, edit the citation format (``Preferences > Better BibTeX``):

- ``Citation keys > Citation key formula``:

  .. code-block:: none

      auth.capitalize + year

- ``Export > Fields > Fields to omit from export``:

  .. code-block:: none

      file, langid, abstract, urldate, copyright, keywords, annotation

- ``Export > Export unicode as plain text latex commands``: uncheck

To update ``references.bib``, export the library as ``Better BibTeX``.

Setting NCBI API key
--------------------
The example gallery as well as the tutorial use :mod:`biotite.database.entrez`
to fetch sequence data.
Hence, these scripts may raise a ``RequestError`` due to
a hight number of requests to the NCBI Entrez database.
This can be fixed by exporting the ``NCBI_API_KEY`` environment variable,
containing an
`NCBI API key <https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/>`_.