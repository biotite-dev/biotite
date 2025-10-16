.. This source code is part of the Biotite package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

Contributor guide
=================

As the aim of *Biotite* is to create a comprehensive library, we welcome
developers who would like to extend the package with new functionalities or
improve existing code.
Contributing new examples to the gallery or improving the documentation
in general is also highly appreciated.

The complete development workflow is hosted on
`GitHub <https://github.com/biotite-dev/biotite>`_.
This is also the place where you would post feature propositions,
questions, bug reports, etc.

If you are interested in improving *Biotite*, you feel free to join our chat on
`Discord <https://discord.gg/cUjDguF>`_.
We are happy to answer questions, discuss ideas and provide mentoring for
newcomers.
Alternatively, you can also contact `<padix.key@gmail.com>`_.
A good place to find projects to start with are the
`Open Issues <https://github.com/biotite-dev/biotite/issues>`_ and
the `Project Boards <https://github.com/biotite-dev/biotite/projects>`_.

The following pages should explain development guidelines in
order to keep *Biotite*'s source code consistent.
Finally, the :doc:`deployment` describes the process of releasing a new
version of *Biotite*.

Requirements
------------

Development of *Biotite* requires a few packages in addition to the ones
specified in
`pyproject.toml <http://raw.githubusercontent.com/biotite-dev/biotite/master/pyproject.toml>`_.
The full list is provided in
`environment.yml <http://raw.githubusercontent.com/biotite-dev/biotite/master/environment.yml>`_.
If you use the `Conda <https://docs.conda.io>`_ package manager, you can simply
create a environment with all required dependencies by running

.. code-block:: console

   $ conda env create -f environment.yml
   $ conda activate biotite-dev

Contributing examples
---------------------

Do you have an application of *Biotite* and you want to share it with the
world?
Then the example gallery is the way to go.
Head directly to the :ref:`gallery section <example_gallery>` to learn how to
contribute.


.. toctree::
    :maxdepth: 1
    :hidden:

    development
    testing
    documentation
    deployment
