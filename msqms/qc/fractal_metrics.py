# -*- coding: utf-8 -*-
"""Fractal Domain Metric for MEG Data."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import antropy as ant
from msqms.qc import Metrics
from msqms.constants import MEG_TYPE
from msqms.utils import segment_raw_data


class FractalDomainMetric(Metrics):
    """
    Class to calculate fractal domain metrics for MEG data.

    This class processes segmented MEG data and computes fractal dimension metrics
    for each segment, similar to entropy domain metrics.

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
        Compute fractal domain metrics for segmented MEG data.

        Parameters
        ----------
        meg_type : MEG_TYPE
            Type of MEG channels to process (e.g., 'mag', 'grad').
        seg_length : int, optional
            Length of each segment for computation, in seconds. Default is 100.

        Returns
        -------
        meg_metrics_df : pd.DataFrame
            DataFrame containing the averaged fractal metrics for the MEG data.
        """
        raw_list, _ = segment_raw_data(self.raw, seg_length)
        meg_metrics_list = [self._compute_fractal_metrics(raw_i, meg_type) for raw_i in raw_list]

        # Combine and average metrics
        combined_metrics = meg_metrics_list[0]
        for metrics in meg_metrics_list[1:]:
            combined_metrics += metrics
        meg_metrics_df = combined_metrics / len(meg_metrics_list)

        return meg_metrics_df

    def _compute_fractal_metrics(self, raw, meg_type: MEG_TYPE):
        """
        Compute all fractal-related metrics for a single MEG segment.

        Parameters
        ----------
        raw : mne.io.Raw
            The raw MEG segment.
        meg_type : MEG_TYPE
            Type of MEG channels to process.

        Returns
        -------
        meg_metrics_df : pd.DataFrame
            DataFrame containing the fractal metrics for the MEG segment.
        """
        self.meg_type = meg_type
        self.meg_names = self._get_meg_names(self.meg_type)
        self.meg_data = raw.get_data(meg_type)

        # Compute fractal metrics
        fractal_metrics = self.compute_fractal_dimension(self.meg_data)

        # Combine and summarize metrics
        meg_metrics_df = fractal_metrics.copy()
        meg_metrics_df.loc[f"avg_{meg_type}"] = meg_metrics_df.mean(axis=0)
        meg_metrics_df.loc[f"std_{meg_type}"] = meg_metrics_df.std(axis=0)

        return meg_metrics_df

    @staticmethod
    def _ant_1d_fractal_dimension(data: np.ndarray):
        """
        Calculate fractal-related features for a single channel.

        Parameters
        ----------
        data : np.ndarray
            Time series data for a single channel.

        Returns
        -------
        metrics : list
            List of fractal-related metrics.
        """
        # Petrosian fractal dimension
        pfd = ant.petrosian_fd(data)

        # Katz fractal dimension
        kfd = ant.katz_fd(data)

        # Higuchi fractal dimension
        hfd = ant.higuchi_fd(data)

        # Detrended fluctuation analysis
        dfa = ant.detrended_fluctuation(data)

        return [pfd, kfd, hfd, dfa]

    def compute_fractal_dimension(self, data: np.ndarray):
        """
        Calculate fractal dimensions for all channels.

        Parameters
        ----------
        data : np.ndarray
            Multichannel time series data.

        Returns
        -------
        fractal_df : pd.DataFrame
            DataFrame containing fractal dimension metrics for all channels.
        """
        single_fractal = Parallel(self.n_jobs)(
            delayed(self._ant_1d_fractal_dimension)(single_ch_data) for single_ch_data in data
        )
        fractal_df = pd.DataFrame(single_fractal,
                                  columns=["PFD", "KFD", "HFD", "DFA"],
                                  index=self.meg_names)
        return fractal_df