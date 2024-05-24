# -*- coding: utf-8 -*-
"""the overview of quality."""
import mne
import numpy as np
from scipy.stats import skew,kurtosis
from opmqc.libs.pyprep.find_noisy_channels import NoisyChannels
from opmqc.utils.logging import clogger
from opmqc.libs.osl import detect_badsegments, detect_badchannels

class QualityOverview(object):
    def __init__(self, raw):
        self.raw = raw
        self.quality_overview_score = None
        self.parameter_config = None
        self.bad_channels = None

        # automatic calculation
        self.data, self.ch_num = self.preprocess()
        stats = self.stats_summary()

        self.mean_values = stats[0]
        self.var_values = stats[1]
        self.std_values = stats[2]
        self.max_values = stats[3]
        self.min_values = stats[4]

    def preprocess(self):
        """ 1.down sampling 100Hz.(for speed up.)
            2.filter 1-40Hz?
            3.HFC?
            Return:
                data (np.array): [channels * times]
        """
        meg_indices = mne.pick_types(raw.info, meg=True)
        data = raw.get_data()[meg_indices]
        ch_num = len(meg_indices)
        return data, ch_num

    # basic stats info
    def stats_summary(self):
        """mean/max/min/std/median average on times"""
        mean_values = np.nanmean(self.data, axis=1)
        var_values = np.nanvar(self.data, axis=1)
        std_values = np.nanstd(self.data, axis=1)
        max_values = np.nanmax(self.data, axis=1)
        min_values = np.nanmin(self.data, axis=1)
        median_values = np.nanmedian(self.data, axis=1)
        return [mean_values, var_values, std_values, max_values, min_values,median_values]

    def stats_skewness(self):
        """
        compute the ratio of left tail of the distributrion by channels.
        compute the mean of skewness.
        """
        skewness = skew(self.data,axis=1,bias=True)
        left_tail = len(skewness[skewness> 0])
        left_skew_ratio = left_tail / self.data.shape[0]
        mean_skewness = np.mean(skewness)
        return left_skew_ratio,mean_skewness

    def stats_kurtosis(self):
        """
        compute mean kurtosis by channel.
        compute the playkurtic ratio by channel.
        """
        kurtosis_value = kurtosis(self.data, axis=1, bias=True)
        playkurtic_ratio = len(kurtosis_value[kurtosis_value < 3]) / self.data.shape[0]
        mean_kurtosis = np.mean(kurtosis_value)
        return mean_kurtosis, playkurtic_ratio

    # analysis over time
    def average_psd_over_time(self):
        pass

    def amplitude_over_time(self):
        pass

    def freq_over_time(self):
        pass

    def find_bad_channels(self):
        noisy_data = NoisyChannels(self.raw, random_state=1337)
        # find bad by corr

        noisy_data.find_bad_by_correlation()
        clogger.info(f"pyprep: finding bad channels by corr.{noisy_data.bad_by_correlation}")

        # find bad by deviation
        noisy_data.find_bad_by_deviation()
        clogger.info(f"pyprep: finding bad channels by deviation.{noisy_data.bad_by_deviation}")

        # find bad by snr
        noisy_data.find_bad_by_SNR()
        clogger.info(f"pyprep: finding bad channels by snr.{noisy_data.bad_by_SNR}")

        # find bad by nan flat
        noisy_data.find_bad_by_nan_flat()
        clogger.info(f"pyprep: finding bad channels by nan:{noisy_data.bad_by_nan}--flat:{noisy_data.bad_by_flat}")

        noisy_data.find_bad_by_hfnoise()
        clogger.info(f"pyprep: finding bad channels by hfonoise.{noisy_data.bad_by_hf_noise}")

        # find bad by ransac
        noisy_data.find_bad_by_ransac(channel_wise=True, max_chunk_size=1)
        clogger.info(f"pyprep: finding bad channels by ransac[slow].{noisy_data.bad_by_ransac}")

        self.bad_channels = noisy_data.get_bads()
        clogger.info(f"Get All Bad Channels:{self.bad_channels}")

    def find_bad_channels_by_psd(self):
        """
        Calculate the PSD(power spectral density) of all channels,
        find the ones that exceed the mean plus standard deviation, and determine them as bad channels.
        """
        ch_names = np.array(self.raw.info['ch_names'])
        psd = raw.compute_psd()
        psd_data = psd.get_data()
        ch_mean_psd = np.mean(psd_data, axis=1)
        total_mean = np.mean(ch_mean_psd)
        total_std = np.std(ch_mean_psd)
        ids = np.where(ch_mean_psd > total_mean + total_std)
        bad_channel = ch_names[ids[0]]
        clogger.info(f"channel name:{bad_channel}")
        return bad_channel

    def find_bad_channels_by_osl(self):
        bad_channel = detect_badchannels(self.raw,picks='mag',ref_meg=False)
        clogger.info(f"channel name:{bad_channel}")
        return bad_channel

    def find_bad_segments_by_osl(self):
        bad_segs = detect_badsegments(self.raw, ref_meg=False)
        clogger.info(f"bad segments:{bad_segs}")
        return bad_segs

    def find_zero_values(self):
        """
        Detect zero values.
        Returns
        -------
            - zero_mask
            - zero_ratio
        """
        zero_mask = np.argwhere(self.data == 0)
        zero_count = len(zero_mask)
        total_elements = self.data.size
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
        nan_mask = np.isnan(self.data)
        nan_count = np.sum(nan_mask)
        total_elements = self.data.size
        nan_ratio = (nan_count / total_elements) * 100
        return nan_mask, nan_ratio

    def find_flat(self):
        """detect flat channels or constant channels."""
        flat_thres = 1e-50  # Need to change to dynamic call
        flat_chan_inds = np.argwhere(self.std_values <= flat_thres)
        flat_chan_names = [self.raw.info['ch_names'][fc[0]] for fc in flat_chan_inds]
        flat_chan_ratio = (len(flat_chan_names) / self.ch_num) * 100  # percentage
        return {"flat_chan_names": flat_chan_ratio,
                "flat_chan_ratio": flat_chan_ratio}

    def find_jumps(self):
        pass

    

if __name__ == '__main__':
    from opmqc.main import test_squid_fif_path, test_opm_fif_path
    from opmqc.main import opm_visual_fif_path

    from mne.io import read_raw_fif
    from opmqc.io import read_raw_mag

    raw = read_raw_fif(test_squid_fif_path,verbose=False)
    qov = QualityOverview(raw)
    # print(qov.mean_values)
    # qov.find_bad_channels()
    qov.find_bad_channels_by_psd()
    qov.find_bad_channels_by_osl()
    qov.find_bad_segments_by_osl()



    raw = read_raw_fif(test_opm_fif_path,verbose=False)
    qov = QualityOverview(raw)
    # print(qov.mean_values)
    # qov.find_bad_channels()
    qov.find_bad_channels_by_psd()
    qov.find_bad_channels_by_osl()
    qov.find_bad_segments_by_osl()