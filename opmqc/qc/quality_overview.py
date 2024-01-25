# -*- coding: utf-8 -*-
"""the overview of quality."""


class QualityOverview(object):
    def __init__(self, raw):
        self.raw = raw

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
        pass

    def find_NaN_values(self):
        pass

    def find_fat(self):
        pass

    def find_jumps(self):
        pass


if __name__ == '__main__':
    from opmqc.main import test_squid_fif_path, test_opm_fif_path
    from mne.io import read_raw_fif

    raw = read_raw_fif(test_opm_fif_path, verbose=False)
    qov = QualityOverview(raw)
