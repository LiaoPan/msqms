# -*- coding: utf-8 -*-
"""the overview of quality."""
import mne
import numpy as np


class QualityOverview(object):
    def __init__(self, raw):
        self.raw = raw
        self.quality_overview_score = None
        self.parameter_config = None

    def _preprocess(self):
        """ 1.down sampling 100Hz.(for speed up.)
            2.filter 1-40Hz?
            3.HFC?
        """
        pass

    # basic stats info
    def stats_summary(self):
        """mean/max/min/std """
        pass

    # analysis over time
    def average_psd_over_time(self):
        pass

    def amplitude_over_time(self):
        pass

    def freq_over_time(self):
        pass

    def find_bad_channels(self):
        pass

    def find_bad_segments(self):
        pass

    def find_zero_values(self):
        """
        Detect zero values.
        Returns
        -------
            - zero_mask
            - zero_ratio
        """
        meg_indices = mne.pick_types(raw.info, meg=True)
        data = raw.get_data()[meg_indices]
        zero_mask = np.argwhere(data == 0)
        zero_count = len(zero_mask)
        total_elements = data.size
        zero_ratio = (zero_count / total_elements) * 100
        return zero_mask, zero_ratio

    def find_NaN_values(self):
        """
            Detect NaN values.
        Returns
        -------
            - NaN mask matrix
            - NaN ratio, accounts for all data points.
        """
        meg_indices = mne.pick_types(raw.info, meg=True)
        data = raw.get_data()[meg_indices]
        nan_mask = np.isnan(data)
        nan_count = np.sum(nan_mask)
        total_elements = data.size
        nan_ratio = (nan_count / total_elements) * 100
        return nan_mask, nan_ratio

    def find_fat(self):
        pass

    def find_jumps(self):
        pass


if __name__ == '__main__':
    from opmqc.main import test_squid_fif_path, test_opm_fif_path
    from mne.io import read_raw_fif
    from opmqc.io import read_raw_mag

    raw = read_raw_fif(test_opm_fif_path, verbose=False)
    qov = QualityOverview(raw)
