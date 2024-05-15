# -*- coding: utf-8 -*-
"""
Abstract class for metrics
"""
from abc import ABC, abstractmethod
import mne
import numpy as np


class Metrics(ABC):
    def __init__(self, raw: mne.io.Raw, n_jobs=-1, verbose=False):
        self.raw = raw
        self.samp_freq = raw.info['sfreq']
        self.meg_names = None
        self.meg_type = None
        self.meg_data = None
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _get_meg_names(self, meg_type: str):
        """
        get channel names from meg type('mag','grad').
        """
        picks = mne.pick_types(self.raw.info, meg_type, ref_meg=False)  #If True include CTF / 4D reference channels(ref_meg).
        self.meg_names = np.array(self.raw.info['ch_names'])[picks]
        return self.meg_names
