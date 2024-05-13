# -*- coding: utf-8 -*-
"""
Abstract class for metrics
"""
from abc import ABC, abstractmethod
import mne
import numpy as np


class Metrics(ABC):
    def __init__(self, raw: mne.io.Raw):
        self.raw = raw
        self.meg_names = None
        self.meg_type = None
        self.meg_data = None

    def _get_meg_names(self, meg_type: str):
        """
        get channel names from meg type('mag','grad').
        """
        picks = mne.pick_types(self.raw.info, meg_type)
        self.meg_names = np.array(self.raw.info['ch_names'])[picks]
        return self.meg_names
