# -*- coding: utf-8 -*-
"""Statistics Domain quality control metric."""
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import annotate_muscle_zscore
from scipy.stats import skew, kurtosis
from msqms.qc import Metrics
from msqms.constants import MEG_TYPE
from msqms.utils import clogger
from msqms.libs.pyprep.find_noisy_channels import NoisyChannels
from msqms.libs.osl import detect_badsegments, detect_badchannels
from msqms.utils import normative_score


class StatsDomainMetric(Metrics):
    """
    Compute statistical domain quality metrics for MEG data.

    Includes baseline offset, skewness, kurtosis, and identification
    of bad channels, segments, and flat signals.
    """

    def __init__(self, raw: mne.io.Raw, data_type, origin_raw: mne.io.Raw = None, n_jobs=-1, verbose=False):
        """
        Initialize the statistical domain metric computation.

        Parameters
        ----------
        raw : mne.io.Raw
            The MEG raw data.
        data_type : str
            Data type of the MEG ('opm' or 'squid').
        origin_raw : mne.io.Raw, optional
            Original raw data for muscle annotation, by default None.
        n_jobs : int, optional
            Number of parallel jobs, by default -1.
        verbose : bool, optional
            Enable verbose logging, by default False.
        """
        super().__init__(raw, n_jobs=n_jobs, data_type=data_type, origin_raw=origin_raw, verbose=verbose)

    def compute_metrics(self, meg_type: MEG_TYPE):
        """
        Compute statistical quality metrics for MEG data(all_channels * all_timepoints).

        Parameters
        ----------
        meg_type : MEG_TYPE
            The MEG channel type ('mag', 'grad', or 'eeg').

        Returns
        -------
        pd.DataFrame
            DataFrame containing average and standard deviation of the metrics.
        """
        meg_metrics = dict()
        self.all_meg_names = self._get_meg_names(True)
        self.all_meg_data = self.raw.get_data('meg')

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

        # Bad channels
        bad_channels_ratio, self.bad_chan_names, self.bad_chan_index, self.bad_chan_mask = self.identify_bad_channels()

        meg_metrics["BadChanRatio"] = bad_channels_ratio
        clogger.info(f"Bad channels: {self.bad_chan_names} (ratio: {bad_channels_ratio})")
        clogger.info(f"Get all bad channel:{self.bad_chan_names}--BadChanRatio:{meg_metrics['BadChanRatio']}")

        # bad segments detection
        bad_segs_num, self.bad_seg_mask = self.find_bad_segments_by_osl()
        bad_segs_thres = float(self.data_type_specific_config['BadSegmentsRatio_threshold'])
        bad_segs_ratio = normative_score(bad_segs_num, bad_segs_thres)
        meg_metrics['BadSegmentsRatio'] = bad_segs_ratio
        clogger.info(f"BadSegmentsRatio is {bad_segs_ratio}")

        # Zero and NaN values
        self.zero_mask, zero_ratio = self.find_zero_values(self.all_meg_data)
        self.nan_mask, nan_ratio = self.find_NaN_values(self.all_meg_data)

        # Flat channels
        flat_thres_det = self.data_type_specific_config['flat_wave_detection_threshold']
        flat_info = self.find_flat(flat_thres_det)
        flat_ratio = flat_info['flat_chan_ratio']
        self.flat_mask = flat_info['flat_chan_mask']

        meg_metrics['Zero_ratio'] = zero_ratio
        meg_metrics['NaN_ratio'] = nan_ratio
        meg_metrics['Flat_chan_ratio'] = flat_ratio

        # average
        meg_metrics_df = pd.DataFrame([meg_metrics], index=[f'avg_{meg_type}'])
        meg_metrics_df.loc[f'std_{meg_type}'] = [0] * len(meg_metrics_df.columns)  # meg_metrics_df.std()

        return meg_metrics_df

    def identify_bad_channels(self):
        """
        Identify bad channels using multiple methods and create a mask.

        Returns
        -------
        float
            Ratio of bad channels.
        list
            List of bad channel names.
        np.ndarray
            Mask indicating bad channels.
        """
        bad_channels = set()

        # Pyprep
        prep_ratio, prep_bad = self.find_bad_channels_by_prep()
        bad_channels.update(prep_bad)

        # PSD
        psd_ratio, psd_bad = self.find_bad_channels_by_psd()
        bad_channels.update(psd_bad)

        # OSL
        osl_ratio, osl_bad = self.find_bad_channels_by_osl()
        bad_channels.update(osl_bad)

        # Create bad channel mask
        bad_chan_names = list(bad_channels)
        bad_chan_index = self._get_channel_index(bad_chan_names)
        bad_chan_mask = np.full(self.all_meg_data.shape, False, dtype=bool)
        bad_chan_mask[bad_chan_index] = True
        print("bad_chan_index:", bad_chan_index, "bad_chan_mask:", bad_chan_mask.shape)

        # Final ratio and list
        ratio = len(bad_channels) / len(self.all_meg_names)
        return ratio, bad_chan_names, bad_chan_index, bad_chan_mask

    def _get_channel_index(self, channel_name_list):
        """Returns the channel index based on the channel name
        """
        ch_index = []
        for i in channel_name_list:
            ch_index.append(self.raw.ch_names.index(i))
        return ch_index

    def compute_skewness(self, data: np.ndarray):
        """ Skewness: measure the shape of the distribution

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
        """Kurtosis:
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
        # noisy_data.find_bad_by_correlation()
        # clogger.info(f"pyprep: finding bad channels by corr.{noisy_data.bad_by_correlation}")

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
        # noisy_data.find_bad_by_ransac(channel_wise=True, max_chunk_size=1)
        # clogger.info(f"pyprep: finding bad channels by ransac[slow].{noisy_data.bad_by_ransac}")

        bad_channels = noisy_data.get_bads()
        clogger.info(f"Get All Bad Channels:{bad_channels}")
        bad_channels_ratio = len(bad_channels) / len(self.all_meg_names)
        return bad_channels_ratio, bad_channels

    def find_bad_channels_by_psd(self):
        """Calculate the PSD (power spectral density) of all channels.
        find the ones that exceed the mean plus 3*standard deviation, and determine them as bad channels.
        """
        ch_names = np.array(self.raw.info['ch_names'])
        psd = self.raw.compute_psd()
        psd_data = psd.get_data()
        ch_mean_psd = np.mean(psd_data, axis=1)
        total_mean = np.mean(ch_mean_psd)
        total_std = np.std(ch_mean_psd)
        ids = np.where((ch_mean_psd > total_mean + 3 * total_std))
        bad_channel = ch_names[ids[0]]
        bad_channels_ratio = len(bad_channel) / len(self.all_meg_names)
        clogger.info(f"Detect BadChannels by PSD: {bad_channel}")
        return bad_channels_ratio, bad_channel

    def find_bad_channels_by_osl(self):
        """Find the bad channels by OSL Library.
        """
        bad_channel_mag = detect_badchannels(self.raw, picks='mag', ref_meg=False)
        try:
            bad_channel_grad = detect_badchannels(self.raw, picks='grad')
        except ValueError:
            bad_channel_grad = []
        bad_channel = bad_channel_mag + bad_channel_grad
        bad_channels_ratio = len(bad_channel) / len(self.all_meg_names)
        clogger.info(
            f"Bad channel name:{bad_channel}--Bad channels ratio:{bad_channels_ratio}--all channels:{len(self.all_meg_names)}")
        return bad_channels_ratio, bad_channel

    def find_bad_segments_by_osl(self):
        bad_segs_num = 0
        annot_muscle = None
        bad_segs_osl = detect_badsegments(self.raw, picks=self.meg_type, ref_meg=False, segment_len=1000,
                                          detect_zeros=False, significance_level=0.05, annotate=False)

        # clogger.info(f"bad segments by osl:{bad_segs_osl['onsets']}")
        # mne
        if self.origin_raw.info['lowpass'] >= 140 and self.origin_raw.info['highpass'] <= 110:
            annot_muscle, _ = annotate_muscle_zscore(self.origin_raw, ch_type=self.meg_type, threshold=5,
                                                     filter_freq=[110, 140])
            # clogger.info(f"bad segments by mne:{annot_muscle.onset}")

        # merge
        if annot_muscle != None:
            osl_onsets = bad_segs_osl['onsets']
            mne_onsets = annot_muscle.onset
            mne_durs = annot_muscle.duration

            tmp_onset = []
            tmp_dur = []
            for idx, o in enumerate(osl_onsets):
                if o not in mne_onsets:
                    tmp_onset.append(o)
                    tmp_dur.append(bad_segs_osl['durations'][idx])
            bad_segs = {"onsets": np.append(mne_onsets, tmp_onset), "durations": np.append(mne_durs, tmp_dur)}
        else:
            bad_segs = bad_segs_osl

        if np.any(bad_segs):
            bad_segs_num = len(bad_segs['onsets'])

        clogger.info(f"bad segments num:{bad_segs_num}")
        bad_seg_mask = np.full(self.meg_data.shape, False, dtype=bool)

        # bad segments mask
        for idx, onset in enumerate(bad_segs['onsets']):
            duration = bad_segs['durations'][idx]
            seg_start = self.raw.time_as_index(onset)[0]
            seg_end = seg_start + int(duration * self.raw.info['sfreq'])
            bad_seg_mask[:, seg_start:seg_end] = True

        return bad_segs_num, bad_seg_mask

    def find_zero_values(self, data: np.ndarray):
        """
        Detect zero values.
        Parameters
        ----------
        data :

        Returns
        -------
        zero_mask: np.ndarray
            the mask of zero values.
        zero_ratio: float
            the ratio of zero values.
        """
        zero_mask_positions = np.argwhere(data == 0)
        zero_mask = np.full(data.shape, False, dtype=bool)
        for pos in zero_mask_positions:
            zero_mask[tuple(pos)] = True

        zero_count = len(zero_mask_positions)
        total_elements = data.size
        zero_ratio = (zero_count / total_elements) * 100
        return zero_mask, zero_ratio

    def find_NaN_values(self, data: np.ndarray):
        """
        Detect NaN values
        Parameters
        ----------
        data :

        Returns
        -------
            - NaN mask matrix
            - NaN ratio, accounts for all data points.
        """
        nan_mask = np.isnan(data)
        nan_count = np.sum(nan_mask)
        # total_elements = data.size
        thres = float(self.data_type_specific_config['NaN_ratio_threshold'])
        nan_ratio = normative_score(nan_count, thres)
        # nan_ratio = (nan_count / total_elements) * 100
        return nan_mask, nan_ratio

    def find_flat(self, flat_thres):
        """detect flat channels or constant channels."""
        if isinstance(flat_thres, str):
            flat_thres = float(flat_thres)
        std_values = np.nanstd(self.all_meg_data, axis=1)
        flat_chan_inds = np.argwhere(std_values <= flat_thres)
        flat_chan_names = [self.raw.info['ch_names'][fc[0]] for fc in flat_chan_inds]
        flat_chan_ratio = (len(flat_chan_names) / len(self.meg_names))  # * 100  # percentage
        flat_chan_mask = np.full(self.all_meg_data.shape, False, dtype=bool)
        for fc in flat_chan_inds:
            flat_chan_mask[fc] = True

        return {"flat_chan_names": flat_chan_names,
                "flat_chan_ratio": flat_chan_ratio,
                "flat_chan_mask": flat_chan_mask}

    def compute_mag_field_change(self, data: np.ndarray):
        """Calculate the Mag Field Change,and record the degree of magnetic field change.
        Calculate the maximum value of the magnetic field change, and the mean value and variance of the magnetic field change by channel.
        """
        diff_field = np.abs(np.diff(data, axis=1))

        max_field_change = np.max(diff_field)
        mean_field_change = np.mean(diff_field)
        std_field_change = np.std(diff_field)
        return max_field_change, mean_field_change, std_field_change

    def compute_baseline_offset(self, data: np.ndarray):
        """Baseline offset: Calculate the baseline drift of each channel (mean, median)；
        Calculate the average deviation degree of the channel data mean relative to the population mean and population median.
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
