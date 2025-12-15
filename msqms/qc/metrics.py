# -*- coding: utf-8 -*-
"""
Abstract class for metrics
"""

from abc import ABC, abstractmethod
import mne
import numpy as np
from typing import Dict
from msqms.utils import get_configure
from msqms.constants import MEG_TYPE


class Metrics(ABC):
    """
    Abstract base class for defining metrics for MEG data quality control.

    This class provides common functionality and structure for specific
    metric implementations, such as entropy, fractal, and frequency domain metrics.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw MEG data object.
    data_type : str
        Type of MEG data, such as 'squid' or 'opm'.
    origin_raw : mne.io.Raw, optional
        The original raw MEG data for comparison. Default is None.
    n_jobs : int, optional
        Number of parallel jobs to use for computation. Default is -1 (use all available cores).
    verbose : bool, optional
        If True, enables verbose output. Default is False.

    Attributes
    ----------
    raw : mne.io.BaseRaw
        The raw MEG data object.
    origin_raw : mne.io.Raw or None
        The original raw MEG data for comparison.
    samp_freq : float
        Sampling frequency of the MEG data.
    meg_names : np.ndarray or None
        Names of selected MEG channels, determined by the `meg_type`.
    meg_type : str
        Type of MEG channels ('mag' or 'grad'). Default is 'mag'.
    meg_data : np.ndarray or None
        Data of selected MEG channels.
    data_type : str
        Type of MEG data, such as 'squid' or 'opm'.
    verbose : bool
        Enables verbose output if True.
    n_jobs : int
        Number of parallel jobs for computation.
    config_default : dict
        Default configuration parameters.
    data_type_specific_config : dict
        Data type-specific configuration parameters.
    zero_mask : np.ndarray or None
        Cached mask for zero-value data points (for reporting purposes).
    nan_mask : np.ndarray or None
        Cached mask for NaN data points (for reporting purposes).
    bad_chan_mask : np.ndarray or None
        Cached mask for bad channels.
    bad_seg_mask : np.ndarray or None
        Cached mask for bad segments.
    flat_mask : np.ndarray or None
        Cached mask for flat-line data points.
    bad_chan_names : list or None
        List of names of bad channels.
    bad_chan_index : list or None
        List of indices of bad channels.
    """

    def __init__(self, raw: mne.io.BaseRaw, data_type, origin_raw: mne.io.Raw = None, n_jobs=-1, verbose=False):
        self.raw = raw
        self.origin_raw = origin_raw
        self.samp_freq = raw.info['sfreq']
        self.meg_names = None
        self.meg_type = 'mag'
        self.meg_data = None
        self.data_type = data_type
        self.verbose = verbose
        self.n_jobs = n_jobs

        if self.data_type == 'squid':
            self.raw.pick(self.meg_type)

        # configure
        config_dict = self.get_configure()
        self.config_default = config_dict['default']
        self.data_type_specific_config = config_dict['data_type']

        # cache variances for report
        self.zero_mask = None
        self.nan_mask = None
        self.bad_chan_mask = None
        self.bad_seg_mask = None
        self.flat_mask = None
        self.bad_chan_names = None
        self.bad_chan_index = None

    def _get_meg_names(self, meg_type: MEG_TYPE | bool) -> np.ndarray:
        """
        Retrieve channel names based on MEG type ('mag' or 'grad').

        Parameters
        ----------
        meg_type : MEG_TYPE
            Type of MEG channels to select ('mag' or 'grad').

        Returns
        -------
        meg_names : np.ndarray
            Names of selected MEG channels.
        """
        picks = mne.pick_types(self.raw.info, meg=meg_type, exclude=[''], ref_meg=False)
        self.meg_names = np.array(self.raw.info['ch_names'])[picks]
        return self.meg_names

    def get_configure(self) -> Dict:
        """
        Retrieve configuration parameters from the configuration file.

        The configuration is specific to the MEG data type ('squid' or 'opm').

        Returns
        -------
        config_dict : dict
            Configuration dictionary containing default and data type-specific parameters.
        """
        config_dict = get_configure(self.data_type)
        return config_dict

    @abstractmethod
    def compute_metrics(self, meg_type: MEG_TYPE):
        """
        Abstract method for computing quality control metrics.

        Subclasses must implement this method to define the specific
        metrics computation for the chosen MEG type.

        Parameters
        ----------
        meg_type : MEG_TYPE
            Type of MEG channels to process ('mag' or 'grad').
        """
        pass
