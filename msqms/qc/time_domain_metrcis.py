# -*- coding: utf-8 -*-
"""Time Domain quality control metric."""
import mne
import numpy as np
import pandas as pd

from msqms.qc import Metrics
from msqms.constants import MEG_TYPE


class TimeDomainMetric(Metrics):
    """
    A class to compute time-domain quality control metrics for MEG signals.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw MEG data.
    data_type : MEG_TYPE
        Type of MEG data (e.g., 'mag' or 'grad').
    origin_raw : mne.io.Raw
        Original raw data for reference.
    n_jobs : int, optional
        Number of jobs to use for parallel processing, by default 1.
    verbose : bool, optional
        Whether to output verbose information, by default False.
    """

    def __init__(self, raw: mne.io.Raw, data_type, origin_raw, n_jobs=1, verbose=False):
        super().__init__(raw, n_jobs=n_jobs, data_type=data_type, origin_raw=origin_raw, verbose=verbose)

    def compute_metrics(self, meg_type: MEG_TYPE):
        """
        Compute time-domain quality metrics for the specified MEG type.

        Parameters
        ----------
        meg_type : MEG_TYPE
            The type of MEG data (e.g., 'mag', 'grad') to compute metrics for.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the computed metrics.
        """
        self.meg_type = meg_type
        self.meg_names = self._get_meg_names(self.meg_type)
        self.meg_data = self.raw.get_data(meg_type)

        time_list = []

        # Mapping dictionary for old variables to new names
        metric_dict = {
            "max_ptp": [],
            "form_factor": [],
            "peak_factor": [],
            "pulse_factor": [],
            "margin_factor": []
        }

        for i in range(self.meg_data.shape[0]):
            max_ptp = self.compute_ptp(self.meg_data[i, :])
            form_factor, peak_factor, pulse_factor, margin_factor = self.compute_1d_factors(self.meg_data[i, :])

            metric_dict["max_ptp"].append(max_ptp)
            metric_dict["form_factor"].append(form_factor)
            metric_dict["peak_factor"].append(peak_factor)
            metric_dict["pulse_factor"].append(pulse_factor)
            metric_dict["margin_factor"].append(margin_factor)

        # Create DataFrame for the computed metrics
        metric_df = pd.DataFrame(metric_dict, index=self.meg_names)
        time_list.append(metric_df)

        # Compute maximum-minimum range (MMR)
        mmr = self.compute_max_min_range(self.meg_data)
        time_list.append(pd.DataFrame({'mmr': mmr}, index=self.meg_names))

        # Compute magnetic field change metrics
        max_field_change, mean_field_change, std_field_change = self.compute_max_field_change(self.meg_data)
        time_list.append(
            pd.DataFrame({'max_field_change': max_field_change, "mean_field_change": mean_field_change,
                          "std_field_change": std_field_change}, index=self.meg_names))

        # Compute root-mean-square (RMS)
        rms = self.compute_rms(self.meg_data)
        time_list.append(pd.DataFrame({'rms': rms}, index=self.meg_names))

        # Compute average rectified value (ARV)
        arv = self.compute_1d_arv(self.meg_data)
        time_list.append(pd.DataFrame({'arv': arv}, index=self.meg_names))

        # Compute statistical summary
        stats_df = self.stats_summary(self.meg_data)
        time_list.append(stats_df)

        # Combine all metrics into one DataFrame
        meg_metrics_df = pd.concat(time_list, axis=1)

        # Add average and standard deviation for each metric
        meg_metrics_df.loc[f'avg_{meg_type}'] = meg_metrics_df.mean(axis=0)
        meg_metrics_df.loc[f'std_{meg_type}'] = meg_metrics_df.std(axis=0)

        return meg_metrics_df

    def stats_summary(self, data: np.ndarray):
        """
        Compute statistical summaries for the data.
        mean/max/min/std/median average on times

        Parameters
        ----------
        data : numpy.ndarray
            Input data of shape (n_channels, n_times).

        Returns
        -------
        pandas.DataFrame
            Statistical summary containing mean, variance, std, max, min, and median values.
        """

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
        """
        Compute the maximum peak-to-peak amplitude.
        Maximum Peak-to-peak | Note that there should be instability in mne's peak_finder algorithm;
        Parameters
        ----------
        data : numpy.ndarray
            1D array of MEG data for a single channel.

        Returns
        -------
        float
            Maximum peak-to-peak amplitude.
        """
        from mne.preprocessing import peak_finder
        peak_loc, peak_mag = peak_finder(data, verbose=False)
        diff_mag_abs = np.abs(np.diff(peak_mag))
        max_ptp = np.max(diff_mag_abs, initial=0)
        return max_ptp

    def compute_max_min_range(self, data: np.ndarray):
        """
        Compute the range of maximum and minimum values for each channel.

        Parameters
        ----------
        data : numpy.ndarray
            2D array of MEG data of shape (n_channels, n_times).

        Returns
        -------
        numpy.ndarray
            Array of max-min range for each channel.
        """
        mmr = np.ptp(data, axis=1)
        return mmr

    def compute_max_field_change(self, data: np.ndarray):
        """
        Calculate the Max Field Change, which measures the extent of magnetic field fluctuations.
        Calculate the maximum value of the magnetic field change, and the mean value and variance of the magnetic field change by channel.

        Parameters
        ----------
        data : numpy.ndarray
            2D array of MEG data of shape (n_channels, n_times).

        Returns
        -------
        tuple
            Tuple containing arrays of max, mean, and std field changes for each channel.
        """
        diff_field = np.abs(np.diff(data, axis=1))

        max_field_change = np.max(diff_field, axis=1)
        mean_field_change = np.mean(diff_field, axis=1)
        std_field_change = np.std(diff_field, axis=1)
        return max_field_change, mean_field_change, std_field_change

    def compute_rms(self, data: np.ndarray):
        """
        Compute root-mean-square (RMS) for each channel.

        Parameters
        ----------
        data : numpy.ndarray
            2D array of MEG data of shape (n_channels, n_times).

        Returns
        -------
        numpy.ndarray
            RMS values for each channel.
        """
        return np.sqrt(np.mean(np.square(data), axis=1))

    def compute_1d_arv(self, data: np.ndarray):
        """
         Compute average rectified value (ARV).

         Parameters
         ----------
         data : numpy.ndarray
             2D array of MEG data of shape (n_channels, n_times).

         Returns
         -------
         numpy.ndarray
             ARV values for each channel.
         """
        return np.mean(np.abs(data), axis=1)

    def compute_1d_factors(self, data):
        """
        Compute signal quality factors, including form factor, peak factor, pulse factor, and margin factor.

        Parameters
        ----------
        data : numpy.ndarray
            1D array of MEG data for a single channel.

        Returns
        -------
        tuple
            A tuple of factors (form_factor, peak_factor, pulse_factor, margin_factor).
        """
        rms_value = np.sqrt(np.mean(np.square(data)))  # Root mean square (RMS)
        arv_value = np.mean(np.abs(data))  # Average rectified value (ARV)
        peak_to_peak_value = np.max(data) - np.min(data)  # Peak-to-peak amplitude
        root_amplitude_mean = np.mean(np.sqrt(np.abs(data)))  # Root amplitude mean

        form_factor = rms_value / arv_value
        peak_factor = peak_to_peak_value / rms_value
        pulse_factor = peak_to_peak_value / arv_value
        margin_factor = peak_to_peak_value / root_amplitude_mean

        # note: S,C,I,L
        return form_factor, peak_factor, pulse_factor, margin_factor