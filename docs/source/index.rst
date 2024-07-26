.. opmqc documentation master file, created by
   sphinx-quickstart on Wed Aug 30 22:11:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to opmqc's documentation!
=================================
`OPMQC <https://github.com/LiaoPan/opmqc>`_ is a fully automated quality control tool for OPM-MEG and SQUID-MEG.

.. image:: /_static/report.png

Features
---------

.. grid::

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Reliability and Robustness
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            OPMQC can achieve a more reliable and consistent data quality distribution based on large MEG datasets.
            This results in quantitative signal quality metrics that are more stable and better suited for extensive MEG research.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Modularity and Integrability
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            OPMQC is implemented in modular fashion and provides command-line tools for generating quality reports, as well as functional interfaces for easy integration into third-party software.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Minimal preprocessing
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            OPMQC workflows should be as minimal as possible to estimate the MSQMs score on the original data or their minimally processed derivatives.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Interoperability and standards
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            OPMQC provides each quality metric along with its reference range,
            offering researchers better interpretability and relatively objective standards.


.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`rocket_launch;2em` Installation
         :class-card: sd-text-black sd-bg-light
         :link: quickstart/installation.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`bolt;2em` Guide for Beginner
         :class-card: sd-text-black sd-bg-light
         :link: quickstart/quick_guide.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`library_books;2em` Concepts
         :class-card: sd-text-black sd-bg-light
         :link: tutorial/concepts.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`terminal;2em` Command Tools
         :class-card: sd-text-black sd-bg-light
         :link: quickstart/cmd.html


   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`webhook;2em` API documentation
         :class-card: sd-text-black sd-bg-light
         :link: apis/opmqc.html

..   .. grid-item::
      :columns: 6 6 6 4

..      .. card:: :material-regular:`settings;2em` Quality Reference
         :class-card: sd-text-black sd-bg-light
         :link: quality_reference.html


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Quickstart

   quickstart/installation
   quickstart/quick_guide.rst
   quickstart/cmd.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   tutorial/concepts.rst
   apis/opmqc.rst
..   tutorial/example_opm.rst
..   tutorial/example_squid.rst





