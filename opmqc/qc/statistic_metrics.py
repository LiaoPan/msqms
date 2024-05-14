# -*- coding: utf-8 -*-
"""Time Domain quality control metric."""
import mne
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.stats import skew, kurtosis
from opmqc.qc import Metrics
from opmqc.constants import MEG_TYPE
from opmqc.utils import clogger
from opmqc.libs.pyprep.find_noisy_channels import NoisyChannels


class StatsDomainMetric(Metrics):
    def __init__(self, raw: mne.io.Raw, n_jobs=1, verbose=False):
        super().__init__(raw, n_jobs=n_jobs, verbose=verbose)

    def compute_stats_metrics(self, meg_type: MEG_TYPE):
        """
        calculate  total meg_data(all_channels * all_timepoints)  quality metrics.
        """
        meg_metrics = dict()
        self.meg_type = meg_type
        self.meg_names = self._get_meg_names(self.meg_type)
        self.meg_data = self.raw.get_data(meg_type)

        max_mean_offset, mean_offset, std_mean_offset, max_median_offset, median_offset, std_median_offset = self.compute_baseline_offset(
            self.meg_data)

        meg_metrics['max_mean_offset'] = max_mean_offset
        meg_metrics['mean_offset'] = mean_offset
        meg_metrics['std_mean_offset'] = std_mean_offset
        meg_metrics['max_median_offset'] = max_median_offset
        meg_metrics['median_offset'] = median_offset
        meg_metrics['std_median_offset'] = std_median_offset

        # bad_ch_ratio_by_pyprep = self.find_bad_channels_by_prep()
        bad_ch_ratio_by_psd = self.find_bad_channels_by_psd()
        # meg_metrics['BadChanRatio_Prep'] = bad_ch_ratio_by_pyprep
        meg_metrics['BadChanRatio_PSD'] = bad_ch_ratio_by_psd

        _, zero_ratio = self.find_zero_values(self.meg_data)
        _, nan_ratio = self.find_NaN_values(self.meg_data)
        flat_thres = 1e-20  # Need to change to dynamic call
        flat_info = self.find_flat(flat_thres)
        flat_ratio = flat_info['flat_chan_ratio']
        meg_metrics['Zero_ratio'] = zero_ratio
        meg_metrics['NaN_ratio'] = nan_ratio
        meg_metrics['Flat_chan_ratio'] = flat_ratio

        # average
        meg_metrics_df = pd.DataFrame([meg_metrics], index=[f'avg_{meg_type}'])
        # print(meg_metrics_df)
        # meg_metrics_df.loc[f'avg_{meg_type}'] = meg_metrics_df.mean()
        meg_metrics_df.loc[f'std_{meg_type}'] = meg_metrics_df.mean()

        return meg_metrics_df

    def compute_skewness(self, data: np.ndarray):
        """
        # 1.计算偏度(Skewness)
        # skewness 衡量分布的shape
        # skewness = 0, normally distributed
        # skewness > 0,  more weight in the left tail of the distribution.
        # skewnees < 0， more weight in the right tail of the distribution.

        compute the ratio of left tail of the distributrion.[by channels]
        compute the mean of skewness.
        """
        skewness = skew(data, axis=1, bias=True)
        left_tail = len(skewness[skewness > 0])
        left_skew_ratio = left_tail / data.shape[0]
        mean_skewness = np.nanmean(skewness)
        std_skewness = np.nanmean(skewness)
        return left_skew_ratio, mean_skewness, std_skewness

    def compute_kurtosis(self, data: np.ndarray):
        """
        # 2.计算峰度(Kurtosis)
        # It is also a statistical term and an important characteristic of frequency distribution.
        # It determines whether a distribution is heavy-tailed in respect of the distribution.
        # It provides information about the shape of a frequency distribution.
        # Kurtosis for normal distribution is equal to 3.
        # For a distribution having kurtosis < 3: It is called playkurtic.
        # For a distribution having kurtosis > 3, It is called leptokurtic
        # and it signifies that it tries to produce more outliers rather than the normal distribution.

        compute mean kurtosis by channel.
        compute the playkurtic ratio by channel.
        """
        kurtosis_value = kurtosis(data, bias=True)
        playkurtic_ratio = len(kurtosis_value[kurtosis_value < 3]) / data.shape[0]
        mean_kurtosis = np.mean(kurtosis_value)
        return mean_kurtosis, playkurtic_ratio

    def find_bad_channels_by_prep(self):
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

        bad_channels = noisy_data.get_bads()
        clogger.info(f"Get All Bad Channels:{bad_channels}")
        bad_channels_ratio = len(bad_channels) / len(self.meg_names)
        return bad_channels_ratio

    def find_bad_channels_by_psd(self):
        """
        Calculate the PSD(power spectral density) of all channels,
        find the ones that exceed the mean plus standard deviation, and determine them as bad channels.
        """
        ch_names = np.array(self.raw.info['ch_names'])
        psd = self.raw.compute_psd()
        psd_data = psd.get_data()
        ch_mean_psd = np.mean(psd_data, axis=1)
        total_mean = np.mean(ch_mean_psd)
        total_std = np.std(ch_mean_psd)
        ids = np.where(ch_mean_psd > total_mean + total_std)
        bad_channel = ch_names[ids[0]]
        bad_channels_ratio = len(bad_channel) / len(self.meg_names)
        return bad_channels_ratio

    def find_zero_values(self, data: np.ndarray):
        """
        Detect zero values.
        Returns
        -------
            - zero_mask
            - zero_ratio
        """
        zero_mask = np.argwhere(data == 0)
        zero_count = len(zero_mask)
        total_elements = data.size
        zero_ratio = (zero_count / total_elements) * 100
        return zero_mask, zero_ratio

    def find_NaN_values(self, data: np.ndarray):
        """
            Detect NaN values.
        Returns
        -------
            - NaN mask matrix
            - NaN ratio, accounts for all data points.
        """
        nan_mask = np.isnan(data)
        nan_count = np.sum(nan_mask)
        total_elements = data.size
        nan_ratio = (nan_count / total_elements) * 100
        return nan_mask, nan_ratio

    def find_flat(self, flat_thres):
        """detect flat channels or constant channels."""
        std_values = np.nanstd(self.meg_data, axis=1)
        flat_chan_inds = np.argwhere(std_values <= flat_thres)
        flat_chan_names = [self.raw.info['ch_names'][fc[0]] for fc in flat_chan_inds]
        flat_chan_ratio = (len(flat_chan_names) / len(self.meg_names)) * 100  # percentage
        return {"flat_chan_names": flat_chan_ratio,
                "flat_chan_ratio": flat_chan_ratio}

    def compute_mag_field_change(self, data: np.ndarray):
        """
        # 5.计算Mag Field Change,记录磁场变化程度
        按通道计算磁场变化的最大值,及其磁场变化平均值和方差
        """
        diff_field = np.abs(np.diff(data, axis=1))

        max_field_change = np.max(diff_field)
        mean_field_change = np.mean(diff_field)
        std_field_change = np.std(diff_field)
        return max_field_change, mean_field_change, std_field_change

    def compute_baseline_offset(self, data: np.ndarray):
        """# 6.baseline offset，计算每个通道基线漂移程度（mean、median）；
        计算通道数据均值相对于总体均值、总体中位数的平均偏移程度
        """
        overall_mean = np.mean(data)
        channel_means = np.mean(data, axis=1)
        mea_offset_abs = np.abs(channel_means - overall_mean)
        mean_offset = np.mean(mea_offset_abs)
        std_mean_offset = np.std(mea_offset_abs)
        max_mean_offset = np.max(mea_offset_abs)

        # median
        overall_median = np.median(data)
        channel_medians = np.median(data, axis=1)
        med_offset_abs = np.abs(channel_medians - overall_median)
        median_offset = np.mean(med_offset_abs)
        std_median_offset = np.std(med_offset_abs)
        max_median_offset = np.max(med_offset_abs)

        return max_mean_offset, mean_offset, std_mean_offset, max_median_offset, median_offset, std_median_offset


if __name__ == '__main__':
    from pathlib import Path

    opm_mag_fif = r"C:\Data\Datasets\OPM-Artifacts\S01.LP.fif"
    opm_raw = mne.io.read_raw(opm_mag_fif, verbose=False, preload=True)
    opm_raw.filter(0, 45).notch_filter([50, 100], verbose=False, n_jobs=-1)

    squid_fif = Path(r"C:\Data\Datasets\MEG_Lab\02_liaopan\231123\run1_tsss.fif")
    squid_raw = mne.io.read_raw_fif(squid_fif, preload=True, verbose=False)
    squid_raw.filter(0, 45).notch_filter([50, 100], verbose=False, n_jobs=-1)

    import time

    st = time.time()
    sdm_opm = StatsDomainMetric(opm_raw.crop(0, 0.5), n_jobs=1)
    sdm_squid = StatsDomainMetric(squid_raw.crop(0, 0.5), n_jobs=1)
    print("opm_data:", sdm_opm.compute_stats_metrics('mag'))
    print("squid_data:", sdm_squid.compute_stats_metrics('mag'))
    et = time.time()
    print("cost time:", et - st)
