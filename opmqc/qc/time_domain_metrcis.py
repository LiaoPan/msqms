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


class TimeDomainMetric(Metrics):
    def __init__(self, raw: mne.io.Raw, n_jobs=1, verbose=False):
        super().__init__(raw, n_jobs=n_jobs, verbose=verbose)

    # @hydra.main(config_path="conf",config_name="config")
    def compute_time_metrics(self, meg_type: MEG_TYPE):
        """
        calculate time domain quality metrics.
        """
        self.meg_type = meg_type
        self.meg_names = self._get_meg_names(self.meg_type)
        self.meg_data = self.raw.get_data(meg_type)

        time_list = []

        me_dict = {"max_ptp": [], "S": [], "C": [], "I": [], "L": []}
        for i in range(self.meg_data.shape[0]):
            max_ptp = self.compute_ptp(self.meg_data[i, :])
            s, c, _i, l = self.compute_1d_factors(self.meg_data[i, :])
            me_dict["max_ptp"].append(max_ptp)
            me_dict["S"].append(s)
            me_dict["C"].append(c)
            me_dict["I"].append(_i)
            me_dict["L"].append(l)

        me_df = pd.DataFrame(me_dict, index=self.meg_names)
        time_list.append(me_df)

        mmr = self.compute_max_min_range(self.meg_data)
        time_list.append(pd.DataFrame({'mmr': mmr}, index=self.meg_names))

        max_field_change, mean_field_change, std_field_change = self.compute_max_field_change(self.meg_data)
        time_list.append(
            pd.DataFrame({'max_field_change': max_field_change, "mean_field_change": mean_field_change,
                          "std_field_change": std_field_change}, index=self.meg_names))

        rms = self.compute_rms(self.meg_data)
        time_list.append(pd.DataFrame({'rms': rms}, index=self.meg_names))

        arv = self.compute_1d_arv(self.meg_data)
        time_list.append(pd.DataFrame({'arv': arv}, index=self.meg_names))

        stats_df = self.stats_summary(self.meg_data)
        time_list.append(stats_df)

        # average
        meg_metrics_df = pd.concat(time_list, axis=1)

        meg_metrics_df.loc[f'avg_time_metrics_{meg_type}'] = meg_metrics_df.mean(axis=0)
        meg_metrics_df.loc[f'std_time_metrics_{meg_type}'] = meg_metrics_df.mean(axis=0)

        return meg_metrics_df

    def stats_summary(self, data: np.ndarray):
        """mean/max/min/std/median average on times"""
        mean_values = np.nanmean(data, axis=1)
        var_values = np.nanvar(data, axis=1)
        std_values = np.nanstd(data, axis=1)
        max_values = np.nanmax(data, axis=1)
        min_values = np.nanmin(data, axis=1)
        median_values = np.nanmedian(data, axis=1)
        stats_df = pd.DataFrame({'mean': mean_values, 'variance': var_values,
                                 "std_values": std_values, "max_values": max_values,
                                 "min_values": min_values, "median_values": median_values}, index=self.meg_names)
        return stats_df

    def compute_ptp(self, data: np.ndarray):
        """
        # 3.最大峰峰值PTP | 如果使用mne的peak_finder,peak_finder算法应该会存在不稳定的地方；
        """
        from mne.preprocessing import peak_finder
        peak_loc, peak_mag = peak_finder(data, verbose=False)
        diff_mag_abs = np.abs(np.diff(peak_mag))
        max_ptp = np.max(diff_mag_abs, initial=0)
        return max_ptp

    def compute_max_min_range(self, data: np.ndarray):
        """
        # 4.计算磁场最大与最小值的范围. range of values(maximum-minium) along an axis.
        按通道计算最大值和最小值的范围，并计算均值和方差。
        """
        mmr = np.ptp(data, axis=1)
        return mmr

    def compute_max_field_change(self, data: np.ndarray):
        """
        # 5.计算Max Field Change,记录磁场变化程度
        按通道计算磁场变化的最大值,及其磁场变化平均值和方差
        """
        diff_field = np.abs(np.diff(data, axis=1))

        max_field_change = np.max(diff_field, axis=1)
        mean_field_change = np.mean(diff_field, axis=1)
        std_field_change = np.std(diff_field, axis=1)
        return max_field_change, mean_field_change, std_field_change


    def compute_rms(self, data: np.ndarray):
        """
        # 7.均方根
        """
        return np.sqrt(np.mean(np.square(data), axis=1))

    def compute_1d_arv(self, data: np.ndarray):
        """ 8. 计算整流平均值
          绝对值的平均值
        """
        return np.mean(np.abs(data), axis=1)

    def compute_1d_factors(self, data):
        """# 单通道
        # 9,10,11,12.计算波形因子(Form factor)、峰值因子、脉冲因子、裕度因子
        # ref: https://zhuanlan.zhihu.com/p/621622520
        """
        signal_rms = np.sqrt(np.mean(np.square(data)))  # 均方根
        signal_arv = np.mean(np.abs(data))  # 整流平均值,Average rectified value
        signal_pk = np.max(data) - np.min(data)  # 峰峰值
        signal_xr = np.mean(np.sqrt(np.abs(data)))

        S = signal_rms / signal_arv  # 波形因子
        C = signal_pk / signal_rms  # 峰值因子
        I = signal_pk / signal_arv  # 脉冲因子
        L = signal_pk / signal_xr  # 裕度因子
        return S, C, I, L


if __name__ == '__main__':
    from pathlib import Path

    opm_mag_fif = r"C:\Data\Datasets\OPM-Artifacts\S01.LP.fif"
    opm_raw = mne.io.read_raw(opm_mag_fif, verbose=False, preload=True)
    # opm_raw.filter(0, 45).notch_filter([50, 100], verbose=False, n_jobs=-1)

    squid_fif = Path(r"C:\Data\Datasets\MEG_Lab\02_liaopan\231123\run1_tsss.fif")
    squid_raw = mne.io.read_raw_fif(squid_fif, preload=True, verbose=False)
    # squid_raw.filter(0, 45).notch_filter([50, 100], verbose=False, n_jobs=-1)

    import time

    st = time.time()
    tdm_opm = TimeDomainMetric(opm_raw.crop(0, 0.5), n_jobs=1)
    tdm_squid = TimeDomainMetric(squid_raw.crop(0, 0.5), n_jobs=1)
    print("opm_data:", tdm_opm.compute_time_metrics('mag'))
    print("squid_data:", tdm_squid.compute_time_metrics('grad'))
    et = time.time()
    print("cost time:", et - st)
