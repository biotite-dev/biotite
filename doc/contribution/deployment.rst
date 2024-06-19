Deployment of a new release
===========================
This section describes how create and deploy a release build of the *Biotite*
package and documentation.
Therefore, this section primarily addresses the maintainers of the project.

CCD update
----------
:mod:`biotite.structure.info` bundles selected information from the
`Chemical Component Dictionary <https://www.wwpdb.org/data/ccd>`_ (CCD).
From time to time, this dataset needs an update to include new components
added to the CCD.
This is achieved by running ``setup_ccd.py``.

Creating a new release
----------------------
When a new *GitHub* release is created, the CI jobs building the distributions
and documentation in ``test_and_deploy.yml`` are triggered.
After the successful completion of these jobs, the artifacts are added to the
release.
The distributions for different platforms and Python versions are automatically
uploaded to *PyPI*.

Conda release
-------------
Some time after the release on GitHub, the ``conda-forge`` bot will also create
an automatic pull request for the new release of the
`Conda package <https://github.com/conda-forge/biotite-feedstock>`_.
If no dependencies changed, this pull request can usually be merged without
further effort.

Documentation website
---------------------
The final step of the deployment is putting the directory containing the built
documentation onto the server hosting the website.

The document root of the website should look like this:

.. code-block::

   ├─ .htaccess
   ├─ latest -> x.y.z/
   ├─ x.y.z/
   │  ├─ index.html
   │  ├─ ...
   ├─ a.b.c/
      ├─ index.html
      ├─ ...

``x.y.z/`` and ``a.b.c/`` represent the documentation directories for two
different *Biotite* release versions.

``.htaccess`` should have the following content:

.. code-block:: apache

   RewriteBase /
   RewriteEngine On
   # Redirect if page name does not start with 'latest' or version identifier
   RewriteRule ^(?!latest|\d+\.\d+\.\d+|robots.txt)(.*) latest/$1 [R=301,L]

   ErrorDocument 404 /latest/404.html