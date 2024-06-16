Guide for Beginner
=============


We can use the :doc:`gen_quality_report <../apis/opmqc.reports.report>` function to generate quality control reports：

.. code-block:: python

    from opmqc.reports import gen_quality_report

    opm_fif_path = "<your_opm_file>"

    gen_quality_report([opm_fif_path], outdir=r"C:\Data\reports", report_fname="report",data_type='opm', ftype='html')

    squid_fif_path = "<your_squid_file>"

    gen_quality_report([opm_fif_path], outdir=r"C:\Data\reports", report_fname="report",data_type='squid', ftype='html')

