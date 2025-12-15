Guide for Beginner
=============


We can use the :doc:`gen_quality_report <../apis/msqms.reports.report>` function to generate quality assessment reports：

By specifying the parameter ``ftype`` as ``html``, we can generate reports in HTML format.
It can also be set to ``json`` to generate reports in JSON format, which is convenient for integration with other third-party applications.
Similarly, the ``megfiles`` parameter can accept either a string or a list, allowing for the generation of quality control reports for multiple MEG files.

.. code-block:: python

    from msqms.reports import gen_quality_report

    # OPM-MEG Report in HTML.
    opm_fif_path = "<your_opm_file>"
    out_dir = "<the path to store the report>"
    report_name = "<the name of report file>"
    meg_quality = gen_quality_report(megfiles=opm_fif_path, outdir=out_dir, report_fname=report_name,data_type='opm', ftype='html')

    # OPM-MEG Report in JSON.
    meg_quality = gen_quality_report(megfiles=opm_fif_path, outdir=out_dir report_fname="report_json",data_type='opm', ftype='json')

    # Reports for multiple MEG files.
    meg_quality = gen_quality_report(megfiles=[opm_fif_path], outdir=out_dir, report_fname=report_name,data_type='opm', ftype='html')

We generate a quality control report on SQUID-MEG data by specifying the parameter ``data_type`` as ``squid``.

.. code-block:: python

    # SQUID-MEG Report in HTML
    squid_fif_path = "<your_squid_file>"
    data_type = "squid"
    meg_quality = gen_quality_report(megfiles=opm_fif_path, outdir=r"C:\Data\reports", report_fname="report_squid",data_type=data_type, ftype='html')

    # SQUID-MEG Report in JSON
    ftype="json"
    meg_quality = gen_quality_report(megfiles=opm_fif_path, outdir=r"C:\Data\reports", report_fname="report_squid",data_type=data_type, ftype=ftype)


How to customize metrics
------------

Using :class:`MetricsFactory`'s :func:`register_custom_metric` method,
we can add our own quality control metrics to the MSQMs quality control score calculations and include these results in the quality control report.


.. code-block:: python

    # -*- coding: utf-8 -*-
    """
    Used to add user-defined metrics.
    """
    import pandas as pd
    from msqms.qc import MetricsFactory
    from msqms.constants import MEG_TYPE
    import numpy as np
    from msqms.reports import gen_quality_report

    short_demo = r"C:\Data\Code\msqms\demo.fif"

    def custom_calc_metric(self, meg_type: MEG_TYPE):
        data = self.raw.get_data(meg_type)

        # main code for custom metric
        custom_metric_name = 'Custom_Metric'
        mean_values = np.nanmean(data, axis=1)

        # get channels names of MEG.
        self.meg_names = self._get_meg_names(self.meg_type)

        # package metric result.
        stats_df = pd.DataFrame({custom_metric_name: mean_values}, index=self.meg_names)
        return stats_df


    # register custom metric
    # The type of quality control metric to which the user-defined metric is added: frequency_domain, time_domain, fractal, entropy,artifacts
    # For example, add custom metric to the Frequency Domain.

    MetricsFactory.register_custom_metric('frequency_domain', custom_calc_metric, custom_metrics_name=['Custom_Metric'])

    # quality report
    gen_quality_report(short_demo, outdir=r"C:\Data\Code\msqms\msqms\reports", data_type='opm',
                       report_fname="demo_report", ftype='html')


Note that you also need to add a reference range for the metric in the ``quality_reference`` folder.
The name ``Custom_Metric`` should be the same as the ``custom_metric_name`` variable in the code above.

``iqr_range`` indicates the reference range of this metric, ``q1`` denotes quartile 1, and ``q3`` denotes quartile 3.
``minimum_l`` and ``maximum_k`` indicate the lower and upper bounds for this quality control metric, respectively.
If the ``bound_threshold_iqr`` value is set in ``conf/<opm or squid>/quality_config.yaml``, the ``iqr_range`` is recalculated based on the ``bound_threshold_iqr`` value.

As follows:
 .. code-block:: yaml

    Custom_Metric:
      iqr_range:
        - 0.0
        - 0.0
      q1: 0.0
      q3: 0.0
      minimum_l: 0
      maximum_k: 0.1


