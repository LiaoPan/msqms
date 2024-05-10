# -*- coding: utf-8 -*-
"""Tsfresh quality control metric."""
import mne
from tsfresh import extract_features
import pandas as pd
import numpy as np

class TsfreshDomainMetric:
    """Tsfresh quality control"""
    def __init__(self,raw:mne.io.Raw):
        self.raw = raw

    def package_meg_df(self,meg_data: np.ndarray, meg_names: list):
        """
         # 封装脑磁数据，构建符合tsfresh的DataFrame数据结构
        Parameters:
            meg_data: channels * times
            meg_names: a list of channel names.
        Return:
         return the dataframe that suited for tefresh packages.
        """
        num_ch = opm_data.shape[0]
        for i in range(num_ch):
            opmdf = pd.DataFrame(opm_data[i, :], columns=['mag_value'])
            opmdf['id'] = meg_names[i]

        return opmdf
    def compute_tsfresh_metrics(self,meg_data: np.ndarray):
        fs = extract_features(opmdf, column_id='id', default_fc_parameters=select_parameters, n_jobs=1)
        return fs