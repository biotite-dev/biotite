Deployment of a new release
===========================
This section describes how to create and deploy a release build of the *Biotite*
package and documentation.
Therefore, this section primarily addresses the maintainers of the project.

Creating a new release
----------------------
When a new *GitHub* release is created, the CI jobs building the distributions
and documentation in ``test_and_deploy.yml`` are triggered.
After the successful completion of these jobs, the artifacts are added to the
release.
The distributions for different platforms and Python versions are automatically
uploaded to *PyPI*.
The documentation is also uploaded to this website via the CI.

Conda release
-------------
Some time after the release on GitHub, the ``conda-forge`` bot will also create
an automatic pull request for the new release of the
`Conda package <https://github.com/conda-forge/biotite-feedstock>`_.
If no dependencies changed, this pull request can usually be merged without
further effort.
