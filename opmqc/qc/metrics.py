# -*- coding: utf-8 -*-
"""
Abstract class for metrics
"""
from abc import ABC, abstractmethod
import mne
import numpy as np
from typing import Dict
from pathlib import Path
from opmqc.utils import read_yaml


class Metrics(ABC):
    def __init__(self, raw: mne.io.Raw, data_type, n_jobs=-1, verbose=False):
        self.raw = raw
        self.samp_freq = raw.info['sfreq']
        self.meg_names = None
        self.meg_type = None
        self.meg_data = None
        self.data_type = data_type
        self.verbose = verbose
        self.n_jobs = n_jobs

        # configure
        config_dict = self.get_configure()
        self.config_default = config_dict['default']
        self.data_type_specific_config = config_dict['data_type']

    def _get_meg_names(self, meg_type: str):
        """
        get channel names from meg type('mag','grad').
        """
        picks = mne.pick_types(self.raw.info, meg_type, ref_meg=False)  #If True include CTF / 4D reference channels(ref_meg).
        self.meg_names = np.array(self.raw.info['ch_names'])[picks]
        return self.meg_names

    def get_configure(self) -> Dict:
        """ get configuration parameters from configuration file[conf folder].
        """
        default_config_fpath = Path(__file__).parent.parent / 'conf' / 'config.yaml'
        if self.data_type == 'opm':
            config_fpath = Path(__file__).parent.parent / 'conf' / 'opm' / 'quality_config.yaml'
        elif self.data_type == 'squid':
            config_fpath = Path(__file__).parent.parent / 'conf' / 'squid' / 'quality_config.yaml'
        else:
            raise ValueError(f'{self.data_type} is not a valid')
        config = read_yaml(config_fpath)
        default_config = read_yaml(default_config_fpath)
        return {'default': default_config, 'data_type': config}