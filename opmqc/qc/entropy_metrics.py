# -*- coding: utf-8 -*-
"""Entropy Domain quality control metric."""
import mne
import pywt
import numpy as np
import antropy as ant
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy
from joblib import Parallel, delayed

from opmqc.qc import Metrics
from opmqc.constants import MEG_TYPE
from opmqc.utils import segment_raw_data


class EntropyDomainMetric(Metrics):
    def __init__(self, raw: mne.io.Raw, data_type, origin_raw, n_jobs=-1, verbose=False):
        super().__init__(raw, n_jobs=n_jobs, data_type=data_type, origin_raw=origin_raw, verbose=verbose)

    def compute_metrics(self, meg_type: MEG_TYPE, seg_length=100):
        raw_list, _ = segment_raw_data(self.raw, seg_length)
        meg_metrics_list = []
        for raw_i in raw_list:
            mdf = self._compute_entropy_metrics(raw_i, meg_type)
            meg_metrics_list.append(mdf)

        # combine and average
        mml = meg_metrics_list[0]
        for i in meg_metrics_list[1:]:
            mml += i
        meg_metrics_df = mml / len(meg_metrics_list)
        return meg_metrics_df

    def _compute_entropy_metrics(self, raw: mne.io.Raw, meg_type: MEG_TYPE):
        """
        compute all entropy metrics
        """
        self.meg_type = meg_type
        self.meg_names = self._get_meg_names(self.meg_type)
        self.meg_data = raw.get_data(meg_type)
        entropy_metrics = self.compute_entropies(self.meg_data)

        fractal_metrics = self.compute_fractal_dimension(self.meg_data)

        psd_entropy_metric = self.compute_psd_entropy(self.meg_data)

        energy_entropy_metric = self.compute_energy_entropy(self.meg_data)
        meg_metrics_df = pd.concat([entropy_metrics, fractal_metrics,
                                    psd_entropy_metric, energy_entropy_metric], axis=1)

        meg_metrics_df.loc[f"avg_{meg_type}"] = meg_metrics_df.mean(axis=0)
        meg_metrics_df.loc[f"std_{meg_type}"] = meg_metrics_df.std(axis=0)

        return meg_metrics_df

    @staticmethod
    def _ant_1d_entropies(data: np.ndarray, samp_freq: float):
        # Permutation entropy
        permutation_entropy = ant.perm_entropy(data, normalize=True)
        # Spectral entropy
        spectral_entropy = ant.spectral_entropy(data, sf=samp_freq, method='welch', normalize=True)
        # Singular value decomposition entropy
        svd_entropy = ant.svd_entropy(data, normalize=True)
        # Approximate entropy
        approximate_entropy = ant.app_entropy(data)
        # Sample entropy
        sample_entropy = ant.sample_entropy(data)
        # Hjorth mobility and complexity
        hjorth_mobility, hjorth_complexity = ant.hjorth_params(data)
        # Number of zero-crossings
        num_of_zero_crossings = ant.num_zerocross(data, normalize=True)
        return [permutation_entropy, spectral_entropy, svd_entropy, approximate_entropy, sample_entropy,
                hjorth_mobility, hjorth_complexity, num_of_zero_crossings]

    def compute_entropies(self, data: np.ndarray):
        """Calculate entropy-related features.
        """
        if self.n_jobs == 1:
            single_entropies = Parallel(self.n_jobs)(
                delayed(self._ant_1d_entropies)(single_ch_data, self.samp_freq) for single_ch_data in data)

            entropy_df = pd.DataFrame(single_entropies,
                                      columns=["permutation_entropy", "spectral_entropy",
                                               "svd_entropy", "approximate_entropy", "sample_entropy",
                                               "hjorth_mobility", "hjorth_complexity", "num_of_zero_crossings"],
                                      index=self.meg_names)
        else:
            single_entropies = Parallel(self.n_jobs)(
                delayed(self._ant_1d_entropies)(single_ch_data, self.samp_freq) for single_ch_data in data)

            entropy_df = pd.DataFrame(single_entropies,
                                      columns=["permutation_entropy", "spectral_entropy",
                                               "svd_entropy", "approximate_entropy", "sample_entropy",
                                               "hjorth_mobility", "hjorth_complexity", "num_of_zero_crossings"],
                                      index=self.meg_names)
        return entropy_df

    # Fractal dimension
    @staticmethod
    def _ant_1d_fractal_dimension(data: np.ndarray):
        """Calculate fractal-related features.
        """
        # Petrosian fractal dimension
        pfd = ant.petrosian_fd(data)
        # Katz fractal dimension
        kfd = ant.katz_fd(data)
        # Higuchi fractal dimension
        hfd = ant.higuchi_fd(data)
        # Detrended fluctuation analysis
        dfa = ant.detrended_fluctuation(data)
        return [pfd, kfd, hfd, dfa]

    def compute_fractal_dimension(self, data: np.ndarray):
        """Calculate the fractal dimension.
        """
        single_fractal = Parallel(self.n_jobs)(
            delayed(self._ant_1d_fractal_dimension)(single_ch_data) for single_ch_data in data)
        fractal_df = pd.DataFrame(single_fractal,
                                  columns=["PFD", "KFD", "HFD", "DFA"],
                                  index=self.meg_names)
        return fractal_df

    @staticmethod
    def _power_spectral_entropy(data, fs):
        freq, psd = welch(data, fs)
        psd_norm = np.divide(psd, psd.sum())
        pse = entropy(psd_norm)
        return pse

    def compute_psd_entropy(self, data: np.ndarray):
        """Multichannel power spectrum entropy
        """
        single_psd_entropy = Parallel(self.n_jobs)(
            delayed(self._power_spectral_entropy)(single_ch_data, self.samp_freq) for single_ch_data in data)
        psd_entropy_df = pd.DataFrame(single_psd_entropy,
                                      columns=["power_spectral_entropy"],
                                      index=self.meg_names
                                      )
        return psd_entropy_df

    @staticmethod
    def _sinch_energy_entropy(data: np.ndarray):
        """Calculate the energy and energy entropy of a single channel, based on the natural index
        """
        Stot = 0  # Total Entropy
        Etot = 0  # Total Energy
        coeffs = pywt.wavedec(data, wavelet='db4', level=5)
        for coef in coeffs:
            energy = np.square(coef)
            energy_ratio = energy / np.sum(energy)
            _entropy = -np.sum(energy_ratio * np.log(energy_ratio))
            Etot += np.sum(energy)
            Stot += _entropy
        ratio = Etot / Stot
        return [Etot, Stot, ratio]

    def compute_energy_entropy(self, data: np.ndarray):
        """Calculate energy and energy entropy by channel。
        """
        single_energy_entropy = Parallel(self.n_jobs)(
            delayed(self._sinch_energy_entropy)(single_ch_data) for single_ch_data in data)
        # mean_energy_entropy = np.mean(single_energy_entropy, axis=0)
        # std_energy_entropy = np.std(single_energy_entropy, axis=0)
        energy_entropy_df = pd.DataFrame(single_energy_entropy,
                                         columns=["Total_Energy", "Total_Entropy", "Energy_Entropy_Ratio"],
                                         index=self.meg_names)

        return energy_entropy_df


if __name__ == '__main__':
    from pathlib import Path

    opm_mag_fif = r"C:\Data\Datasets\OPM-Artifacts\S01.LP.fif"
    opm_raw = mne.io.read_raw(opm_mag_fif, verbose=False, preload=True)
    opm_raw.filter(0, 45).notch_filter([50, 100], verbose=False, n_jobs=-1)

    # squid_fif = Path(r"C:\Data\Datasets\MEG_Lab\02_liaopan\231123\run1_tsss.fif")
    # squid_raw = mne.io.read_raw_fif(squid_fif, preload=True, verbose=False)
    # squid_raw.filter(0, 45).notch_filter([50, 100], verbose=False, n_jobs=-1)

    import time

    st = time.time()
    print("opm_raw：", opm_raw)
    edm_opm = EntropyDomainMetric(opm_raw, n_jobs=8)
    # edm_squid = EntropyDomainMetric(squid_raw.crop(0,0.5),n_jobs=1)
    opm = edm_opm.compute_entropy_metrics('mag')
    print("opm_data:", opm.head(3))
    # suqid = edm_squid.compute_entropy_metrics('grad')
    # print("squid_data:", suqid.head(3))
    et = time.time()
    print("cost time:", et - st)
