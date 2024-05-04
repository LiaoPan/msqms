# -*- coding: utf-8 -*-
"""Visual Inspection"""


class VisualInspection(object):
    def __init__(self, raw):
        self.raw = raw

    def plot_psd(self):
        pass

    def plot_average_psd(self):
        """
        Power Spectral Density average on time.
        Returns
        -------

        """
        pass

    def plot_power_on_ts(self):
        pass

    def plot_gfp(self):

        pass

    def plot_chan_variance_ts(self):
        """
        channel variance time series.
        Returns
        -------

        """

    def plot_average_freq(self):
        """
        Returns
        -------

        """
        pass

    def plot_constant_dist(self):
        """
        constant value time series.
        Returns
        -------

        """

    def plot_bad_channel_topo(self):
        """
        The bad channels topomap.
        Returns
        -------
        """
        pass

    def plot_bad_channel_dist(self):
        """
        The bad channels distribution.
        Returns
        -------
        """
        pass

    def plot_bad_segment_dist(self):
        pass

    def plot_bad_trails_dist(self):
        pass

    def plot_no_signals(self):
        pass

    def plot_high_amplitude(self):
        pass

    def plot_low_freq_and_high_amp(self):
        """
        Low frequency and high amplitude.
        Returns
        -------
        """
        pass



