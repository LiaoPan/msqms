# -*- coding: utf-8 -*-
"""frequency domian quality control metric."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mne
from pathlib import Path
from joblib import Parallel, delayed
from opmqc.utils import clogger
import multiprocessing
from typing import Literal
from opmqc.constants import MEG_TYPE
from opmqc.qc import Metrics
class FreqDomainMetrics(Metrics):
    """frequency domian quality control"""

    def __init__(self, raw: mne.io.Raw):
        super().__init__(raw)

    def _get_fre_domain_features(self, signal, Fs=1000):
        L = len(signal)
        y = abs(np.fft.fft(signal / L))[: int(L / 2)]
        y[0] = 0
        f = np.fft.fftfreq(L, 1 / Fs)[: int(L / 2)]
        fre_line_num = len(y)
        p1 = y.mean()  # 频谱均值
        p2 = np.sqrt(np.sum((y - p1) ** 2) / fre_line_num)  # 频谱均方根值
        p3 = np.sum((y - p1) ** 3) / (fre_line_num * p2 ** 3)
        p4 = np.sum((y - p1) ** 4) / (fre_line_num * p2 ** 4)
        p5 = np.sum(f * y) / np.sum(y)  # 频率重心
        p6 = np.sqrt(np.sum((f - p5) ** 2 * y) / fre_line_num)
        p7 = np.sqrt(np.sum(f ** 2 * y) / np.sum(y))  # 均方根频率
        p8 = np.sqrt(np.sum(f ** 4 * y) / np.sum(f ** 2 * y))
        p9 = np.sum(f ** 2 * y) / np.sqrt(np.sum(y) * np.sum(f ** 4 * y))
        p10 = p6 / p5
        p11 = np.sum((f - p5) ** 3 * y) / (p6 ** 3 * fre_line_num)
        p12 = np.sum((f - p5) ** 4 * y) / (p6 ** 4 * fre_line_num)
        p13 = np.sum(abs(f - p5) * y) / (np.sqrt(p6) * fre_line_num)  # 标准差频率
        p = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13]
        return p

    def compute_freq_features_parallel(self, n_jobs=1):  # hell: 多核心引起BrokenProcessPool问题
        """ 并行版本
        """
        clogger.info("Parallel cores: {}".format(n_jobs))
        Ps = Parallel(n_jobs)(delayed(self._get_fre_domain_features)(single_ch_data) for single_ch_data in self.meg_data)
        mean_freq_feat = np.mean(Ps, axis=0)
        std_freq_feat = np.std(Ps, axis=0)
        return mean_freq_feat, std_freq_feat

    def compute_freq_features(self, meg_type: MEG_TYPE):
        """串行版本的频域特征计算"""

        self.meg_type = meg_type
        self.meg_data = self.raw.get_data(meg_type)

        Ps = []
        for i in range(self.meg_data.shape[0]):
            p = self._get_fre_domain_features(self.meg_data[i, :])
            Ps.append(p)

        mean_freq_feat = np.mean(np.array(Ps), axis=0)
        std_freq_feat = np.std(np.array(Ps), axis=0)
        return mean_freq_feat, std_freq_feat


if __name__ == '__main__':
    opm_mag_fif = "C:\Data\Datasets\OPM-Artifacts\S01.LP.fif"
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
    fdm_opm = FreqDomainMetrics(opm_raw)
    fdm_squid = FreqDomainMetrics(squid_raw)
    print("opm_data:", fdm_opm.compute_freq_features(meg_type='mag'))
    print("squid_data:", fdm_squid.compute_freq_features(meg_type='grad'))
    et = time.time()
    print("cost time:", et - st)
