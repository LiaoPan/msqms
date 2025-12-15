# -*- coding: utf-8 -*-
"""Entropy Domain Metric for MEG Data."""
import pywt
import numpy as np
import pandas as pd
import antropy as ant
from joblib import Parallel, delayed

from msqms.qc import Metrics
from msqms.constants import MEG_TYPE
from msqms.utils import segment_raw_data


class EntropyDomainMetric(Metrics):
    """
    Class to calculate entropy domain metrics for MEG data.

    This class processes segmented MEG data and computes entropy-related metrics
    for each segment.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw MEG data.
    data_type : str
        The type of MEG data (e.g., 'opm' or 'squid').
    origin_raw : mne.io.Raw
        The original raw MEG data for comparison.
    n_jobs : int, optional
        Number of parallel jobs to use for computation. Default is -1 (use all available cores).
    verbose : bool, optional
        If True, enables verbose output. Default is False.
    """

    def __init__(self, raw, data_type, origin_raw, n_jobs=-1, verbose=False):
        super().__init__(raw, n_jobs=n_jobs, data_type=data_type, origin_raw=origin_raw, verbose=verbose)

    def compute_metrics(self, meg_type: MEG_TYPE, seg_length=100):
        """
        Compute entropy domain metrics for segmented MEG data.

        Parameters
        ----------
        meg_type : MEG_TYPE
            Type of MEG channels to process (e.g., 'mag', 'grad').
        seg_length : int, optional
            Length of each segment for computation, in seconds. Default is 100.

        Returns
        -------
        meg_metrics_df : pd.DataFrame
            DataFrame containing the averaged entropy metrics for the MEG data.
        """
        raw_list, _ = segment_raw_data(self.raw, seg_length)
        meg_metrics_list = [self._compute_entropy_metrics(raw_i, meg_type) for raw_i in raw_list]

        # Combine and average metrics
        combined_metrics = meg_metrics_list[0]
        for metrics in meg_metrics_list[1:]:
            combined_metrics += metrics
        meg_metrics_df = combined_metrics / len(meg_metrics_list)

        return meg_metrics_df

    def _compute_entropy_metrics(self, raw, meg_type: MEG_TYPE):
        """
        Compute all entropy-related metrics for a single MEG segment.

        Parameters
        ----------
        raw : mne.io.Raw
            The raw MEG segment.
        meg_type : MEG_TYPE
            Type of MEG channels to process.

        Returns
        -------
        meg_metrics_df : pd.DataFrame
            DataFrame containing the entropy metrics for the MEG segment.
        """
        self.meg_type = meg_type
        self.meg_names = self._get_meg_names(self.meg_type)
        self.meg_data = raw.get_data(meg_type)

        # Compute entropy metrics
        entropy_metrics = self.compute_entropies(self.meg_data)
        energy_entropy_metric = self.compute_energy_entropy(self.meg_data)

        # Combine and summarize metrics
        meg_metrics_df = pd.concat([entropy_metrics, energy_entropy_metric], axis=1)
        meg_metrics_df.loc[f"avg_{meg_type}"] = meg_metrics_df.mean(axis=0)
        meg_metrics_df.loc[f"std_{meg_type}"] = meg_metrics_df.std(axis=0)

        return meg_metrics_df

    @staticmethod
    def _ant_1d_entropies(data: np.ndarray, samp_freq: float):
        """
        Compute one-dimensional entropy metrics for a single channel.

        Parameters
        ----------
        data : np.ndarray
            Time series data for a single channel.
        samp_freq : float
            Sampling frequency of the data.

        Returns
        -------
        metrics : list
            List of entropy-related metrics.
        """
        # Permutation entropy
        permutation_entropy = ant.perm_entropy(data, normalize=True)
        # Spectral entropy
        spectral_entropy = ant.spectral_entropy(data, sf=samp_freq, method='welch', normalize=True)
        # Singular value decomposition entropy
        svd_entropy = ant.svd_entropy(data, normalize=True)
        # Hjorth mobility and complexity
        hjorth_mobility, hjorth_complexity = ant.hjorth_params(data)

        return [permutation_entropy, spectral_entropy, svd_entropy, hjorth_mobility, hjorth_complexity]

    def compute_entropies(self, data: np.ndarray):
        """
        Calculate entropy-related features for all channels.

        Parameters
        ----------
        data : np.ndarray
            Multichannel time series data.

        Returns
        -------
        entropy_df : pd.DataFrame
            DataFrame containing entropy-related metrics for all channels.
        """
        single_entropies = Parallel(self.n_jobs)(
            delayed(self._ant_1d_entropies)(single_ch_data, self.samp_freq) for single_ch_data in data
        )
        entropy_df = pd.DataFrame(single_entropies,
                                  columns=["permutation_entropy", "spectral_entropy",
                                           "svd_entropy", "hjorth_mobility", "hjorth_complexity"],
                                  index=self.meg_names)
        return entropy_df

    @staticmethod
    def _sinch_energy_entropy(data: np.ndarray):
        """
        Compute energy and energy entropy for a single channel.

        Parameters
        ----------
        data : np.ndarray
            Time series data for a single channel.

        Returns
        -------
        metrics : list
            List containing total energy, total entropy, and energy-entropy ratio.
        """
        Stot, Etot = 0, 0  # Total entropy and Total energy
        coeffs = pywt.wavedec(data, wavelet='db4', level=5)
        for coef in coeffs:
            energy = np.square(coef)
            energy_ratio = energy / np.sum(energy)
            _entropy = -np.sum(energy_ratio * np.log(energy_ratio))
            Etot += np.sum(energy)
            Stot += _entropy
        ratio = Etot / Stot

        return [Etot, Stot, ratio]

    def compute_energy_entropy(self, data: np.ndarray):
        """
        Compute energy and energy entropy for all channels.

        Parameters
        ----------
        data : np.ndarray
            Multichannel time series data.

        Returns
        -------
        energy_entropy_df : pd.DataFrame
            DataFrame containing energy and energy entropy metrics for all channels.
        """
        single_energy_entropy = Parallel(self.n_jobs)(
            delayed(self._sinch_energy_entropy)(single_ch_data) for single_ch_data in data
        )
        energy_entropy_df = pd.DataFrame(single_energy_entropy,
                                         columns=["Total_Energy", "Total_Entropy", "Energy_Entropy_Ratio"],
                                         index=self.meg_names)
        return energy_entropy_df
