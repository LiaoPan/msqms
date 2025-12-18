# -*- coding: utf-8 -*-
"""Frequency Domain Metric for MEG Data.
"""

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from msqms.utils import clogger
from msqms.constants import MEG_TYPE
from msqms.qc import Metrics


class FreqDomainMetric(Metrics):
    """
    Class to calculate frequency domain metrics for MEG data.

    This class processes MEG data and computes frequency domain features
    for all MEG channels, with support for both sequential and parallel computation.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw MEG data.
    data_type : str
        The type of MEG data (e.g., 'opm' or 'squid').
    origin_raw : mne.io.Raw
        The original raw MEG data for comparison.
    n_jobs : int, optional
        Number of parallel jobs to use for computation. Default is 1 (no parallelization).
    verbose : bool, optional
        If True, enables verbose output. Default is False.
    """
    def __init__(self, raw: mne.io.Raw, data_type, origin_raw, n_jobs=1, verbose=False):
        super().__init__(raw, n_jobs=n_jobs, data_type=data_type, origin_raw=origin_raw, verbose=verbose)

    @staticmethod
    def _get_fre_domain_features(signal: np.ndarray, Fs=1000) -> dict:
        """
        Compute frequency domain features for a single channel.

        Parameters
        ----------
        signal : np.ndarray
            Time series data for a single channel.
        Fs : float, optional
            Sampling frequency of the data. Default is 1000 Hz.

        Returns
        -------
        features : dict
            A dictionary of computed frequency domain features:
            - mean_amplitude(p1)
            - std_amplitude(p2)
            - skewness_amplitude(p3)
            - kurtosis_amplitude(p4)
            - mean_frequency(p5)
            - std_frequency(p6)
            - rms_frequency(p7)
            - fourth_moment_frequency(p8)
            - normalized_second_moment(p9)
            - frequency_dispersion(p10)
            - frequency_skewness(p11)
            - frequency_kurtosis(p12)
            - mean_absolute_deviation(p13)
        """
        L = len(signal)
        y = abs(np.fft.fft(signal / L))[: int(L / 2)]
        y[0] = 0  # Remove DC component
        f = np.fft.fftfreq(L, 1 / Fs)[: int(L / 2)]
        fre_line_num = len(y)

        # Pre-compute common values to avoid repeated calculations and handle edge cases
        y_mean = y.mean()
        y_var = np.var(y)
        y_std = np.sqrt(y_var) if y_var > 0 else 0.0
        y_sum = np.sum(y)
        
        # Handle zero sum case (flat signal or all zeros after DC removal)
        if y_sum == 0 or y_sum < np.finfo(float).eps:
            # Return default values for flat/zero signals
            return {
                "mean_amplitude": 0.0,
                "std_amplitude": 0.0,
                "skewness_amplitude": 0.0,
                "kurtosis_amplitude": 0.0,
                "mean_frequency": 0.0,
                "std_frequency": 0.0,
                "rms_frequency": 0.0,
                "fourth_moment_frequency": 0.0,
                "normalized_second_moment": 0.0,
                "frequency_dispersion": 0.0,
                "frequency_skewness": 0.0,
                "frequency_kurtosis": 0.0,
                "mean_absolute_deviation": 0.0,
            }
        
        # Compute mean frequency
        mean_freq = np.sum(f * y) / y_sum
        
        # Compute frequency variance and standard deviation
        freq_var = np.sum((f - mean_freq) ** 2 * y) / fre_line_num
        freq_std = np.sqrt(freq_var) if freq_var > 0 else 0.0
        
        # Compute sum of f^2 * y and f^4 * y for later use
        f2y_sum = np.sum(f ** 2 * y)
        f4y_sum = np.sum(f ** 4 * y)
        
        # Handle division by zero for f2y_sum
        if f2y_sum == 0 or f2y_sum < np.finfo(float).eps:
            fourth_moment_freq = 0.0
        else:
            fourth_moment_freq = np.sqrt(f4y_sum / f2y_sum)
        
        # Handle division by zero for normalized_second_moment
        if f4y_sum == 0 or f4y_sum < np.finfo(float).eps:
            normalized_second_moment = 0.0
        else:
            normalized_second_moment = f2y_sum / np.sqrt(y_sum * f4y_sum)

        features = {
            "mean_amplitude": y_mean,
            "std_amplitude": np.sqrt(np.sum((y - y_mean) ** 2) / fre_line_num) if fre_line_num > 0 else 0.0,
            "skewness_amplitude": (np.sum((y - y_mean) ** 3) / (fre_line_num * y_std ** 3)) 
                                   if y_std > 0 and fre_line_num > 0 else 0.0,
            "kurtosis_amplitude": (np.sum((y - y_mean) ** 4) / (fre_line_num * y_std ** 4)) 
                                   if y_std > 0 and fre_line_num > 0 else 0.0,
            "mean_frequency": mean_freq,
            "std_frequency": freq_std,
            "rms_frequency": np.sqrt(f2y_sum / y_sum) if y_sum > 0 else 0.0,
            "fourth_moment_frequency": fourth_moment_freq,
            "normalized_second_moment": normalized_second_moment,
            "frequency_dispersion": (freq_std / mean_freq) if mean_freq != 0 else 0.0,
            "frequency_skewness": (np.sum((f - mean_freq) ** 3 * y) / (freq_std ** 3 * fre_line_num)) 
                                  if freq_std > 0 and fre_line_num > 0 else 0.0,
            "frequency_kurtosis": (np.sum((f - mean_freq) ** 4 * y) / (freq_std ** 4 * fre_line_num)) 
                                 if freq_std > 0 and fre_line_num > 0 else 0.0,
            "mean_absolute_deviation": (np.sum(abs(f - mean_freq) * y) / (freq_std * fre_line_num)) 
                                       if freq_std > 0 and fre_line_num > 0 else 0.0,
        }

        return features

    def compute_metrics(self, meg_type: MEG_TYPE) -> pd.DataFrame:
        """
        Compute frequency domain metrics for MEG data.

        Parameters
        ----------
        meg_type : MEG_TYPE
            Type of MEG channels to process (e.g., 'mag', 'grad').

        Returns
        -------
        freq_feat_df : pd.DataFrame
            DataFrame containing frequency domain metrics for all channels,
            including their average and standard deviation.
        """
        self.meg_type = meg_type
        self.meg_data = self.raw.get_data(meg_type)
        self.meg_names = self._get_meg_names(meg_type)

        if self.n_jobs == 1:
            # Sequential computation
            freq_list = []
            for i in range(self.meg_data.shape[0]):
                features = self._get_fre_domain_features(self.meg_data[i, :], Fs=self.samp_freq)
                freq_list.append(pd.DataFrame([features], index=[self.meg_names[i]]))

            freq_feat_df = pd.concat(freq_list)
        else:
            # Parallel computation
            clogger.info(f"Using {self.n_jobs} parallel cores.")
            # Store samp_freq in a local variable to ensure it's properly captured in the closure
            samp_freq = self.samp_freq
            freq_list = Parallel(self.n_jobs, verbose=10)(
                delayed(self._get_fre_domain_features)(single_ch_data, Fs=samp_freq)
                for single_ch_data in self.meg_data
            )
            freq_feat_df = pd.DataFrame(freq_list, index=self.meg_names)

        # Compute average and standard deviation
        avg_freq_feat = freq_feat_df.mean(axis=0)
        std_freq_feat = freq_feat_df.std(axis=0)
        freq_feat_df.loc[f'avg_{meg_type}'] = avg_freq_feat
        freq_feat_df.loc[f'std_{meg_type}'] = std_freq_feat

        return freq_feat_df
