# -*- coding: utf-8 -*-
"""
Used to add user-defined metrics.
"""
import mne
import pandas as pd
from opmqc.qc import MetricsFactory
from opmqc.constants import MEG_TYPE
from opmqc.qc.msqm import MSQM
from opmqc.main import test_opm_fif_path
import numpy as np
from opmqc.reports import gen_quality_report

short_demo = r"C:\Data\Code\opmqc\demo.fif"


def custom_calc_metric(self, meg_type: MEG_TYPE):
    data = self.raw.get_data(meg_type)

    # main code
    custom_metric_name = 'Custom_Metric'
    mean_values = np.nanmean(data, axis=1)

    self.meg_names = self._get_meg_names(self.meg_type)
    stats_df = pd.DataFrame({custom_metric_name: mean_values}, index=self.meg_names)
    return stats_df


# register custom metric
# frequency_domain、time_domain、fractal、entropy
# Note that you need to add a reference range for the metric in the `quality_reference` folder.
MetricsFactory.register_custom_metric('frequency_domain', custom_calc_metric, custom_metrics_name=['Custom_Metric'])
print("MetricsFactory:", MetricsFactory, id(MetricsFactory))
gen_quality_report([short_demo], outdir=r"C:\Data\Code\opmqc\opmqc\reports", data_type='opm',
                   report_fname="new_demo_report", ftype='html')