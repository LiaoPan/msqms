Command Line Tools
=============

We provide a command-line tool that, after the installation of OPMQC,
can be used to evaluate the quality of single or multiple magnetic brain files and generate a quality control report.

:doc:`See in details <../apis/opmqc.cli.workflow>`

.. code-block:: bash

     # for windows
     $ opmqc_report.exe -f .\auditory_raw.fif -o .\ -t opm

     # for linux
     $ opmqc_report -f ./auditory_raw.fif -o ./ -t opm

See help documents:

.. code-block:: bash

     # for windows
     $ opmqc_report.exe -h

     # for linux
     $ opmqc_report -h

