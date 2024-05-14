# -*- coding: utf-8 -*-
"""Tsfresh quality control metric."""
import mne
from tsfresh import extract_features
import pandas as pd
import numpy as np
from typing import Literal
from opmqc.constants import MEG_TYPE
from opmqc.qc import Metrics

SELECT_PARAMETERS = {'sum_values': None,
                     'abs_energy': None,
                     'mean_abs_change': None,
                     'mean_change': None,
                     'mean_second_derivative_central': None,
                     'median': None,
                     'mean': None,
                     'standard_deviation': None,
                     'variation_coefficient': None,
                     'variance': None,
                     'skewness': None,
                     'kurtosis': None,
                     'root_mean_square': None,
                     'absolute_sum_of_changes': None,
                     'count_above_mean': None,
                     'count_below_mean': None,
                     'percentage_of_reoccurring_values_to_all_values': None,
                     'percentage_of_reoccurring_datapoints_to_all_datapoints': None,
                     'sample_entropy': None,
                     'maximum': None,
                     'absolute_maximum': None,
                     'minimum': None,
                     'benford_correlation': None,
                     'time_reversal_asymmetry_statistic': [{'lag': 1}, {'lag': 2}, {'lag': 3}],
                     'c3': [{'lag': 1}, {'lag': 2}, {'lag': 3}],
                     'cid_ce': [{'normalize': True}],
                     'partial_autocorrelation': [{'lag': 0},
                                                 {'lag': 1},
                                                 {'lag': 2},
                                                 {'lag': 3},
                                                 {'lag': 4},
                                                 {'lag': 5},
                                                 {'lag': 6},
                                                 {'lag': 7},
                                                 {'lag': 8},
                                                 {'lag': 9}],
                     'binned_entropy': [{'max_bins': 10}],
                     'spkt_welch_density': [{'coeff': 2}, {'coeff': 5}, {'coeff': 8}],
                     'fft_aggregated': [{'aggtype': 'centroid'},
                                        {'aggtype': 'variance'},
                                        {'aggtype': 'skew'},
                                        {'aggtype': 'kurtosis'}],
                     'approximate_entropy': [{'m': 2, 'r': 0.5}],
                     'max_langevin_fixed_point': [{'m': 3, 'r': 30}],
                     'augmented_dickey_fuller': [{'attr': 'teststat'},
                                                 {'attr': 'pvalue'},
                                                 {'attr': 'usedlag'}],
                     'number_crossing_m': [{'m': 0}, {'m': -1}, {'m': 1}],
                     'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 0},
                                                {'num_segments': 10, 'segment_focus': 1},
                                                {'num_segments': 10, 'segment_focus': 2},
                                                {'num_segments': 10, 'segment_focus': 3},
                                                {'num_segments': 10, 'segment_focus': 4},
                                                {'num_segments': 10, 'segment_focus': 5},
                                                {'num_segments': 10, 'segment_focus': 6},
                                                {'num_segments': 10, 'segment_focus': 7},
                                                {'num_segments': 10, 'segment_focus': 8},
                                                {'num_segments': 10, 'segment_focus': 9}],
                     'ratio_beyond_r_sigma': [{'r': 0.5},
                                              {'r': 1},
                                              {'r': 1.5},
                                              {'r': 2},
                                              {'r': 2.5},
                                              {'r': 3}],
                     'count_above': [{'t': 0}],
                     'count_below': [{'t': 0}],
                     'lempel_ziv_complexity': [{'bins': 2},
                                               {'bins': 3},
                                               {'bins': 5},
                                               {'bins': 10},
                                               {'bins': 100}],
                     'fourier_entropy': [{'bins': 2},
                                         {'bins': 3},
                                         {'bins': 5},
                                         {'bins': 10},
                                         {'bins': 100}],
                     'permutation_entropy': [{'tau': 1, 'dimension': 3},
                                             {'tau': 1, 'dimension': 4},
                                             {'tau': 1, 'dimension': 5},
                                             {'tau': 1, 'dimension': 6},
                                             {'tau': 1, 'dimension': 7}],
                     'mean_n_absolute_max': [{'number_of_maxima': 7}],

                     }


class TsfreshDomainMetric(Metrics):
    """Tsfresh quality control metrcis"""

    def __init__(self, raw: mne.io.Raw, n_jobs=1, verbose=False):
        super().__init__(raw,n_jobs=n_jobs, verbose=verbose)
        self.select_parameters = SELECT_PARAMETERS  # a list of channel names.

    # def _get_meg_names(self,meg_type:str):
    #     """
    #     get channel names from meg type('mag','grad').
    #     """
    #     picks = mne.pick_types(self.raw.info, meg_type)
    #     meg_names = np.array(self.raw.info['ch_names'])[picks]
    #     return meg_names

    def package_meg_df(self, meg_data: np.ndarray, meg_names: np.ndarray):
        """
        Parameters:
            meg_data: channels * times
            meg_names: a list of channel names.
        Return:
         return the dataframe that suited for tefresh packages.
        """
        num_ch = meg_data.shape[0]
        print("data shape:", meg_data.shape[0], meg_data.shape[1])
        meg_datas_list = []
        for i in range(num_ch):
            opmdf = pd.DataFrame(meg_data[i, :], columns=['mag_value'])
            opmdf['id'] = meg_names[i]
            meg_datas_list.append(opmdf)
        meg_df = pd.concat(meg_datas_list)
        return meg_df


    def compute_tsfresh_metrics(self, meg_type: MEG_TYPE):
        """
        main function for computing tsfresh metrics.
        """
        self.meg_data = self.raw.get_data(meg_type)  # meg_data: channels * times
        self.meg_names = self._get_meg_names(meg_type)

        meg_df = self.package_meg_df(self.meg_data, self.meg_names)
        fs = extract_features(meg_df, column_id='id', default_fc_parameters=self.select_parameters, n_jobs=1)
        fs.loc[f"avg_{meg_type}"] = fs.mean(axis=0)
        fs.loc[f"std_{meg_type}"] = fs.std(axis=0)
        return fs


if __name__ == '__main__':
    from pathlib import Path

    opm_mag_fif = r"C:\Data\Datasets\OPM-Artifacts\S01.LP.fif"
    opm_raw = mne.io.read_raw(opm_mag_fif, verbose=False, preload=True)
    opm_raw.filter(0, 45).notch_filter([50, 100], verbose=False, n_jobs=8)

    squid_fif = Path(r"C:\Data\Datasets\MEG_Lab\02_liaopan\231123\run1_tsss.fif")
    squid_raw = mne.io.read_raw_fif(squid_fif, preload=True, verbose=False)
    squid_raw.filter(0, 45).notch_filter([50, 100], verbose=False, n_jobs=8)

    import time

    st = time.time()
    tfdm_opm = TsfreshDomainMetric(opm_raw.copy().crop(0,0.5))
    print("Debug info：",opm_raw.copy().crop(0,0.5))
    tfdm_squid = TsfreshDomainMetric(squid_raw.copy().crop(0,0.5))
    print("opm_data:", tfdm_opm.compute_tsfresh_metrics('mag'))
    print("squid_data:", tfdm_squid.compute_tsfresh_metrics('mag'))
    et = time.time()
    print("cost time:", et - st)
