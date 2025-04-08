.. This source code is part of the Biotite package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

:html_theme.sidebar_secondary.remove:
:sd_hide_title: true

.. raw:: html

    <style>
        .bd-main .bd-content .bd-article-container {
            /* Wider page */
            max-width: 80rem;
        }
        h1 {
            font-size: 48px;
            text-align: center;
            margin-top: 6rem;
            margin-bottom: 2rem;
        }
        /* Make the hero h/v centered in the window */
        .hero {
            align-items: center;
            justify-content: center;
            display: flex;
            /* Due to navigation bar not the complete window height is required */
            min-height: 80vh;
        }
        /* Avoid truncated hero section on small mobile screens */
        @media screen and (max-width: 768px) {
            .hero {
                display: block;
            }
        }
        /* Limit the size of the sponsor logos */
        .sponsors {
            width: 25%;
        }

    </style>


#####################
Biotite documentation
#####################

.. raw:: html

    <div class="hero">

.. grid:: 1 1 2 2

    .. grid-item::

        .. image:: /static/assets/general/biotite_logo.svg
            :alt: Biotite
            :class: no-scaled-link
            :width: 50%
            :align: center

        .. raw:: html

            <h2>Your multitool for bioinformatics</h2>

        Whether you want to find homologous sequence regions in a protein
        family or would like to identify secondary structure elements in a
        protein structure:
        *Biotite* has the right tool for you.
        This package bundles popular tasks in computational molecular biology
        into a uniform and fast *Python* library.

        Skip writing code for basic functionality and focus on what your code
        makes unique -
        from small analysis scripts to entire bioinformatics software packages.



        .. grid:: 2 3 2 3

            .. grid-item::

                .. button-ref:: tutorial/index
                    :color: primary
                    :expand:
                    :shadow:

                    Get started

            .. grid-item::

                .. button-ref:: examples/index
                    :color: primary
                    :outline:
                    :expand:
                    :shadow:

                    More examples

    .. grid-item::

       .. grid:: 1
           :gutter: 2

           .. grid-item-card::
               :link: examples/gallery/sequence/homology/lexa_conservation.html
               :img-background: /examples/gallery/sequence/homology/images/sphx_glr_lexa_conservation_002.png

           .. grid-item-card::
               :link: examples/gallery/structure/protein/transketolase_sse.html
               :img-background: /examples/gallery/structure/protein/images/sphx_glr_transketolase_sse_004.png

.. raw:: html

    </div>


.. raw:: html

    <h1>Features</h1>

.. grid:: 1 1 2 2
    :gutter: 5

    .. grid-item::

        .. raw:: html

            <h2>Analyze sequence data</h2>

        Work with sequences of any kind: from the usual nucleotide and protein
        sequences to structural alphabets like *3Di*.
        Use the rapid and modular alignment tools to identify homologous
        regions or to map reads.
        Eventually, visualize your results in different *Matplotlib* based
        representations, ranging from sequence alignments to feature maps.

        .. card::
            :img-background: /examples/gallery/sequence/homology/images/sphx_glr_avidin_alignment_001.png
            :link: examples/gallery/sequence/homology/avidin_alignment.html

    .. grid-item::

        .. raw:: html

            <h2>Explore molecular 3D structures</h2>

        Handle 3D structures of large biomolecules as well as small molecules
        in an intuitive *NumPy*-like way.
        Explore a variety of functions to filter, transform and analyze your
        structure data - from surface area calculation to structure
        superimposition.
        Interfaces to a multitude of popular file formats such as PDB,
        CIF/BinaryCIF, MOL/SDF, trajectory formats and more are available.


        .. card::
            :img-background: /examples/gallery/structure/contacts/images/sphx_glr_contact_sites_001.png
            :link: examples/gallery/structure/contacts/contact_sites.html

    .. grid-item::

        .. raw:: html

            <h2>Access data in biological databases</h2>

        Obtain data from biological sequence and structure databases,
        such as *NCBI Entrez*, *UniProt*, *PDB* and *PubChem*.
        Craft your queries with the help of *Pythonic* logical operators
        instead of learning the respective REST API.

        .. card::
            :img-background: /examples/gallery/structure/misc/images/sphx_glr_pdb_statistics_001.png
            :link: examples/gallery/structure/misc/pdb_statistics.html

    .. grid-item::

        .. raw:: html

            <h2>Integrate popular software seamlessly</h2>

        In case *Biotite*'s integrated functionality is not sufficient for your
        tasks, you can use interfaces to prominent external software ranging
        from multiple sequence alignment to secondary structure annotation
        tools.
        These interfaces are seamless:
        You can input *Python* objects and you get *Python* objects.
        File creation and command line execution are handled under the hood.

        .. card::
            :img-background: /examples/gallery/structure/modeling/images/sphx_glr_docking_002.png
            :link: examples/gallery/structure/modeling/docking.html


.. raw:: html

    <h1>Support</h1>

.. grid:: 1 1 2 2
    :gutter: 5

    .. grid-item::

        .. raw:: html

            <h2>Contributors</h2>

        .. raw:: html

            <a href="https://github.com/biotite-dev/biotite/graphs/contributors" style="margin-top: 20px; margin-bottom: 20px;">
                <img src="https://contrib.rocks/image?repo=biotite-dev/biotite&columns=10&max=40" />
            </a>

        Interested in contributing to the project?

        .. button-ref:: contribution/index
            :color: primary
            :outline:
            :shadow:

            More information

    .. grid-item::

        .. raw:: html

            <h2>Citation</h2>

        If you use *Biotite* in a scientific publication, please cite one of the
        following articles:

        .. bibliography::
            :list: bullet

            Kunzmann2018
            Kunzmann2023


.. raw:: html

    <h1>Sponsors</h1>

.. grid::
    :gutter: 5
    :padding: 5
    :class-container: sponsors

    .. grid-item-card::
        :img-background: /static/assets/sponsors/vantai_logo.jpg
        :link: https://www.vant.ai/


.. toctree::
    :maxdepth: 1
    :hidden:

    tutorial/index
    install
    apidoc/index
    examples/index
    extensions
    contribution/index
    logo