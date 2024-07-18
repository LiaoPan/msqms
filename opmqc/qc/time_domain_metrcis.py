# -*- coding: utf-8 -*-
"""Time Domain quality control metric."""
import mne
import numpy as np
import pandas as pd

from opmqc.qc import Metrics
from opmqc.constants import MEG_TYPE


class TimeDomainMetric(Metrics):
    def __init__(self, raw: mne.io.Raw, data_type, origin_raw, n_jobs=1, verbose=False):
        super().__init__(raw, n_jobs=n_jobs, data_type=data_type, origin_raw=origin_raw, verbose=verbose)

    def compute_metrics(self, meg_type: MEG_TYPE):
        """
        calculate time domain quality metrics.
        """
        self.meg_type = meg_type
        self.meg_names = self._get_meg_names(self.meg_type)
        self.meg_data = self.raw.get_data(meg_type)

        time_list = []

        me_dict = {"max_ptp": [], "S": [], "C": [], "I": [], "L": []}
        for i in range(self.meg_data.shape[0]):
            max_ptp = self.compute_ptp(self.meg_data[i, :])
            s, c, _i, l = self.compute_1d_factors(self.meg_data[i, :])
            me_dict["max_ptp"].append(max_ptp)
            me_dict["S"].append(s)
            me_dict["C"].append(c)
            me_dict["I"].append(_i)
            me_dict["L"].append(l)

        me_df = pd.DataFrame(me_dict, index=self.meg_names)
        time_list.append(me_df)

        mmr = self.compute_max_min_range(self.meg_data)
        time_list.append(pd.DataFrame({'mmr': mmr}, index=self.meg_names))

        max_field_change, mean_field_change, std_field_change = self.compute_max_field_change(self.meg_data)
        time_list.append(
            pd.DataFrame({'max_field_change': max_field_change, "mean_field_change": mean_field_change,
                          "std_field_change": std_field_change}, index=self.meg_names))

        rms = self.compute_rms(self.meg_data)
        time_list.append(pd.DataFrame({'rms': rms}, index=self.meg_names))

        arv = self.compute_1d_arv(self.meg_data)
        time_list.append(pd.DataFrame({'arv': arv}, index=self.meg_names))

        stats_df = self.stats_summary(self.meg_data)
        time_list.append(stats_df)

        # average
        meg_metrics_df = pd.concat(time_list, axis=1)

        meg_metrics_df.loc[f'avg_{meg_type}'] = meg_metrics_df.mean(axis=0)
        meg_metrics_df.loc[f'std_{meg_type}'] = meg_metrics_df.std(axis=0)

        return meg_metrics_df

    def stats_summary(self, data: np.ndarray):
        """mean/max/min/std/median average on times"""
        mean_values = np.nanmean(data, axis=1)
        var_values = np.nanvar(data, axis=1)
        std_values = np.nanstd(data, axis=1)
        max_values = np.nanmax(data, axis=1)
        min_values = np.nanmin(data, axis=1)
        median_values = np.nanmedian(data, axis=1)
        stats_df = pd.DataFrame({'mean': mean_values, 'variance': var_values,
                                 "std_values": std_values, "max_values": max_values,
                                 "min_values": min_values, "median_values": median_values}, index=self.meg_names)
        return stats_df

    def compute_ptp(self, data: np.ndarray):
        """Maximum Peak-to-peak | Note that there should be instability in mne's peak_finder algorithm;
        """
        from mne.preprocessing import peak_finder
        peak_loc, peak_mag = peak_finder(data, verbose=False)
        diff_mag_abs = np.abs(np.diff(peak_mag))
        max_ptp = np.max(diff_mag_abs, initial=0)
        return max_ptp

    def compute_max_min_range(self, data: np.ndarray):
        """Calculate the range of maximum and minimum values by channel.
        """
        mmr = np.ptp(data, axis=1)
        return mmr

    def compute_max_field_change(self, data: np.ndarray):
        """Calculate the Max Field Change, which measures the extent of magnetic field fluctuations.
            Calculate the maximum value of the magnetic field change, and the mean value and variance of the magnetic field change by channel.
        """
        diff_field = np.abs(np.diff(data, axis=1))

        max_field_change = np.max(diff_field, axis=1)
        mean_field_change = np.mean(diff_field, axis=1)
        std_field_change = np.std(diff_field, axis=1)
        return max_field_change, mean_field_change, std_field_change

    def compute_rms(self, data: np.ndarray):
        """root-mean-square
        """
        return np.sqrt(np.mean(np.square(data), axis=1))

    def compute_1d_arv(self, data: np.ndarray):
        """ Average rectified value
        """
        return np.mean(np.abs(data), axis=1)

    def compute_1d_factors(self, data):
        """factors calculation。
        """
        signal_rms = np.sqrt(np.mean(np.square(data)))  # root-mean-square
        signal_arv = np.mean(np.abs(data))  # Average rectified value
        signal_pk = np.max(data) - np.min(data)  # peak-to-peak
        signal_xr = np.mean(np.sqrt(np.abs(data)))

        S = signal_rms / signal_arv  # form factor
        C = signal_pk / signal_rms  # peak factor
        I = signal_pk / signal_arv  # pulse factor
        L = signal_pk / signal_xr  # margin factor
        return S, C, I, L
