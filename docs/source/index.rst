.. opmqc documentation master file, created by
   sphinx-quickstart on Wed Aug 30 22:11:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to opmqc's documentation!
=================================
`OPMQC <https://github.com/LiaoPan/opmqc>`_ is a fully automated quality control tool for OPM-MEG.


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

            OPMQC supports object-oriented transformations, including
            JIT compilation, Autograd.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Modularity and Integrability
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            OPMQC provides various numerical integration methods for ODEs, SDEs, DDEs, FDEs, etc.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Model Building
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            OPMQC provides a modular and composable programming interface for building dynamics.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Model Simulation
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            OPMQC supports dynamics simulation for various brain objects with parallel supports.


   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Model Training
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            OPMQC supports dynamics training with various machine learning algorithms, like FORCE learning, ridge regression, back-propagation, etc.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Model Analysis
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            OPMQC supports dynamics analysis for low- and high-dimensional systems, including phase plane, bifurcation, linearization, and fixed/slow point analysis.

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
         :link: tutorials.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`library_books;2em` Core Concepts
         :class-card: sd-text-black sd-bg-light
         :link: core_concepts.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`menu_book;2em` Tutorials for OPM-MEG
         :class-card: sd-text-black sd-bg-light
         :link: tutorials.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`token;2em` Tutorials for SQUID-MEG
         :class-card: sd-text-black sd-bg-light
         :link: advanced_tutorials.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`terminal;2em` Command Tools
         :class-card: sd-text-black sd-bg-light
         :link: toolboxes.html


   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`webhook;2em` API documentation
         :class-card: sd-text-black sd-bg-light
         :link: apis/opmqc.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`settings;2em` Quality Reference
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

   tutorial/core_concepts.rst
   tutorial/example_opm.rst
   tutorial/example_squid.rst
   apis/opmqc.rst

