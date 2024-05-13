# -*- coding: utf-8 -*-
"""Time Domain quality control metric."""
import mne
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from opmqc.qc import Metrics
from opmqc.constants import MEG_TYPE
class TimeDomainMetric(Metrics):
    def __init__(self, raw: mne.io.Raw):
        super().__init__(raw)
    def compute_time_metrics(self,meg_type: MEG_TYPE):
        """
        calculate time domain quality metrics.
        """
        meg_metrics = dict()
        self.meg_type = meg_type
        self.meg_names = self._get_meg_names(self.meg_type)
        self.meg_data = self.raw.get_data(meg_type)

        left_skew_ratio,mean_skewness,std_skewness = self.compute_skewness(self.meg_data)
        meg_metrics['left_skew_ratio'] = left_skew_ratio
        meg_metrics['mean_skewness'] = mean_skewness
        meg_metrics['std_skewness'] = std_skewness

        mean_kurtosis,std_kurtosis, playkurtic_ratio = self.compute_kurtosis(self.meg_data)
        meg_metrics['playkurtic_ratio'] = playkurtic_ratio
        meg_metrics['mean_kurtosis'] = mean_kurtosis
        meg_metrics['std_kurtosis'] = std_kurtosis

        max_ptp, mean_ptp, std_ptp = self.compute_ptp(self.meg_data)
        meg_metrics['max_ptp'] = max_ptp
        meg_metrics['mean_ptp'] = mean_ptp
        meg_metrics['std_ptp'] = std_ptp

        mmr_mean, mmr_std = self.compute_max_min_range(self.meg_data)
        meg_metrics['mmr_mean'] = mmr_mean
        meg_metrics['mmr_std'] = mmr_std

        max_field_change, mean_field_change, std_field_change = self.compute_max_field_change(self.meg_data)
        meg_metrics['max_field_change'] = max_field_change
        meg_metrics['mean_field_change'] = mean_field_change
        meg_metrics['std_field_change'] = std_field_change

        max_mean_offset, mean_offset, std_mean_offset, max_median_offset, median_offset, std_median_offset = self.compute_baseline_offset(self.meg_data)
        meg_metrics['max_mean_offset'] = max_mean_offset
        meg_metrics['mean_offset'] = mean_offset
        meg_metrics['std_mean_offset'] = std_mean_offset
        meg_metrics['max_median_offset'] = max_median_offset
        meg_metrics['median_offset'] = median_offset
        meg_metrics['std_median_offset'] = std_median_offset

        rms = self.compute_rms(self.meg_data)
        meg_metrics['rms'] = rms

        mean_arv, std_arv = self.compute_arv(self.meg_data)
        meg_metrics['mean_arv'] = mean_arv
        meg_metrics['std_arv'] = std_arv

        factors = self.compute_factors(self.meg_data)
        meg_metrics['mean_S'] = factors['S'][0]
        meg_metrics['std_S'] = factors['S'][1]
        meg_metrics['mean_C'] = factors['C'][0]
        meg_metrics['std_C'] = factors['C'][1]
        meg_metrics['mean_I'] = factors['I'][0]













        # self.compute_arv()



        return
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
        return left_skew_ratio, mean_skewness,std_skewness

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
        kurtosis_value = kurtosis(data, axis=1, bias=True)
        playkurtic_ratio = len(kurtosis_value[kurtosis_value < 3]) / data.shape[0]
        mean_kurtosis = np.nanmean(kurtosis_value)
        std_kurtosis = np.nanmean(kurtosis_value)
        return mean_kurtosis,std_kurtosis,playkurtic_ratio

    def compute_ptp(self, data: np.ndarray):
        """
        # 3.最大峰峰值PTP和最小峰峰值PTP | 如果使用mne的peak_finder,peak_finder算法应该会存在不稳定的地方；
        """
        from mne.preprocessing import peak_finder
        n_chan = data.shape[0]
        max_ptps = []
        # min_ptps = []
        diff_mags = []
        for i in range(n_chan):
            "for-code,slowly."
            peak_loc, peak_mag = peak_finder(data[i, :])
            diff_mag_abs = np.abs(np.diff(peak_mag))
            max_diff = np.max(diff_mag_abs, initial=0)
            # min_diff = np.min(diff_mag_abs,initial=0)
            max_ptps.append(max_diff)
            # min_ptps.append(min_diff)
            diff_mags.extend(diff_mag_abs)
        max_ptp = np.max(max_ptps)
        # min_ptp = np.min(min_ptps)
        mean_ptp = np.mean(diff_mags)
        std_ptp = np.std(diff_mags)
        return max_ptp, mean_ptp, std_ptp

    def compute_ptp_parallel(self, data: np.ndarray, n_jobs=-1):
        """并行版本
        # 3*.最大峰峰值PTP和最小峰峰值PTP | 如果使用mne的peak_finder,peak_finder算法应该会存在不稳定的地方；
        """
        from mne.preprocessing import peak_finder
        from joblib import Parallel, delayed
        n_chan = data.shape[0]
        max_ptps = []
        min_ptps = []
        diff_mag_abs = []
        results = Parallel(n_jobs)(delayed(peak_finder)(single_ch_data) for single_ch_data in data)
        for i in results:
            diff_abs = np.abs(np.diff(i[1]))
            diff_mag_abs.extend(diff_abs)
            max_ptps.append(np.max(diff_abs, initial=0))
            # min_ptps.append(np.min(diff_abs,initial=0))
        max_ptp = np.max(max_ptps)
        # min_ptp = np.min(min_ptps)
        mean_ptp = np.mean(diff_mag_abs)
        std_ptp = np.std(diff_mag_abs)
        return max_ptp, mean_ptp, std_ptp

    def compute_max_min_range(self,data: np.ndarray):
        """
        # 4.计算磁场最大与最小值的范围. range of values(maximum-minium) along an axis.
        按通道计算最大值和最小值的范围，并计算均值和方差。
        """
        mmr = np.ptp(data, axis=1)
        mmr_mean = np.mean(mmr)
        mmr_std = np.std(mmr)
        return mmr_mean, mmr_std

    def compute_max_field_change(self,data: np.ndarray):
        """
        # 5.计算Max Field Change,记录磁场变化程度
        按通道计算磁场变化的最大值,及其磁场变化平均值和方差
        """
        diff_field = np.abs(np.diff(data, axis=1))
        # min_field_change = np.min(diff_field)
        max_field_change = np.max(diff_field)
        mean_field_change = np.mean(diff_field)
        std_field_change = np.std(diff_field)
        return max_field_change, mean_field_change, std_field_change

    def compute_baseline_offset(self,data: np.ndarray):
        """
         # 6.baseline offset，计算每个通道基线漂移程度（mean、median）；
        计算通道数据均值相对于总体均值、总体中位数的平均偏移程度
        """
        overall_mean = np.mean(data)
        channel_means = np.mean(data, axis=0)
        mea_offset_abs = np.abs(channel_means - overall_mean)
        mean_offset = np.mean(mea_offset_abs)
        std_mean_offset = np.std(mea_offset_abs)
        max_mean_offset = np.max(mea_offset_abs)

        # median
        overall_median = np.median(data)
        channel_medians = np.median(data, axis=0)
        med_offset_abs = np.abs(channel_medians - overall_median)
        median_offset = np.mean(med_offset_abs)
        std_median_offset = np.std(med_offset_abs)
        max_median_offset = np.max(med_offset_abs)

        return max_mean_offset, mean_offset, std_mean_offset, max_median_offset, median_offset, std_median_offset

    def compute_rms(self,data: np.ndarray):
        """
        # 7.均方根
        """
        return np.sqrt(np.mean(np.square(data)))

    def _compute_1d_arv(self, data: np.ndarray):
        """绝对值的平均值
        """
        return np.mean(np.abs(data))

    def compute_arv(self, data: np.ndarray):
        """# 8. 计算整流平均值
        ## 信号绝对值的平均值
        """
        arv = []
        for i in range(data.shape[0]):
            arv.append(self._compute_1d_arv((data[i, :])))
        mean_arv = np.nanmean(arv)
        std_arv = np.nanstd(arv)
        return mean_arv, std_arv


    def _compute_1d_factors(self, data):
        """# 单通道
        # 9,10,11,12.计算波形因子(Form factor)、峰值因子、脉冲因子、裕度因子
        # ref: https://zhuanlan.zhihu.com/p/621622520
        """
        signal_rms = self.compute_rms(data)  # 均方根
        signal_arv = np.mean(np.abs(data))  # 整流平均值,Average rectified value
        signal_pk = np.max(data) - np.min(data)  # 峰峰值
        signal_xr = np.mean(np.sqrt(np.abs(data)))

        S = signal_rms / signal_arv  # 波形因子
        C = signal_pk / signal_rms  # 峰值因子
        I = signal_pk / signal_arv  # 脉冲因子
        L = signal_pk / signal_xr  # 裕度因子
        return S, C, I, L

    def compute_factors(self, data: np.ndarray):
        """
        9,10,11,12.计算波形因子(Form factor)、峰值因子、脉冲因子、裕度因子
        """
        from joblib import Parallel, delayed
        Ss = []
        Cs = []
        Is = []
        Ls = []
        for i in range(data.shape[0]):
            s, c, i, l = self._compute_1d_factors(data[i, :])
            Ss.append(s)
            Cs.append(c)
            Is.append(i)
            Ls.append(l)
        mean_S, std_S = np.mean(Ss), np.std(Ss)
        mean_C, std_C = np.mean(Cs), np.std(Cs)
        mean_I, std_I = np.mean(Is), np.std(Is)
        mean_L, std_L = np.mean(Ls), np.std(Ls)
        factors = {"S": (mean_S, std_S), "C": {mean_C, std_C}, "I": (mean_I, std_I), "L": (mean_L, std_L)}
        return factors

if __name__ == '__main__':
    from pathlib import Path

    opm_mag_fif = r"C:\Data\Datasets\OPM-Artifacts\S01.LP.fif"
    opm_raw = mne.io.read_raw(opm_mag_fif, verbose=False, preload=True)
    opm_raw.filter(0, 45).notch_filter([50, 100], verbose=False, n_jobs=-1)

    squid_fif = Path(r"C:\Data\Datasets\MEG_Lab\02_liaopan\231123\run1_tsss.fif")
    squid_raw = mne.io.read_raw_fif(squid_fif, preload=True, verbose=False)
    squid_raw.filter(0, 45).notch_filter([50, 100], verbose=False, n_jobs=-1)

    opm_data = opm_raw.get_data('mag')
    squid_data = squid_raw.get_data('mag')
    print("opm_data shape:", opm_data.shape)
    print("squid_data shape:", squid_data.shape)

    import time

    st = time.time()
    tdm_opm = TimeDomainMetric(opm_raw)
    tdm_squid = TimeDomainMetric(squid_raw)
    print("opm_data:", tdm_opm.compute_time_metrics('mag'))
    print("squid_data:", tdm_squid.compute_time_metrics('mag'))
    et = time.time()
    print("cost time:", et - st)