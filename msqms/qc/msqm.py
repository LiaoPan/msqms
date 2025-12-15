# -*- coding: utf-8 -*-
"""
MEG quality assessment based on MEG Signal Quality Metrics(MSQMs)
"""
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from msqms.qc.time_domain_metrcis import TimeDomainMetric
from msqms.qc.freq_domain_metrics import FreqDomainMetric
from msqms.qc.statistic_metrics import StatsDomainMetric
from msqms.qc.entropy_metrics import EntropyDomainMetric
from msqms.qc.fractal_metrics import FractalDomainMetric
from msqms.utils import read_yaml, clogger
from msqms.qc.metrics_factory import MetricsFactory
from msqms.constants import DATA_TYPE, METRICS_COLUMNS, METRICS_DOMAIN

# Register metrics classes based on METRICS_DOMAIN
# ["time_domain", "freq_domain", "stats_domain", "entropy_domain", "custom_domain"]
for domain in METRICS_DOMAIN:
    METRICS_CLASS_MAPPING = {
        "time_domain": TimeDomainMetric,
        "freq_domain": FreqDomainMetric,
        "stats_domain": StatsDomainMetric,
        "entropy_domain": EntropyDomainMetric,
        "fractal_domain": FractalDomainMetric
    }

    metric_class = METRICS_CLASS_MAPPING.get(domain)
    if metric_class:
        MetricsFactory.register_metric(domain, metric_class)


class MSQM:
    """
    MEG Signal Quality Metrics (MSQMs) for assessing the quality of MEG signals.
    """

    def __init__(self, raw: mne.io.Raw, data_type: DATA_TYPE, origin_raw: mne.io.Raw = None, n_jobs=-1, verbose=False):
        """
        Initialize MSQM instance.

        Parameters
        ----------
        raw : mne.io.Raw
            The MEG raw data object.
        data_type : DATA_TYPE
            The type of MEG data ('opm' or 'squid').
        origin_raw : mne.io.Raw, optional
            The original MEG raw data for comparison, by default None.
        n_jobs : int, optional
            Number of parallel jobs for computation, by default -1 (all available cores).
        verbose : bool, optional
            Whether to enable verbose logging, by default False.
        """
        self.raw = raw
        self.origin_raw = origin_raw
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.data_type = data_type
        self.meg_type = 'mag'

        # quality reference ranges
        self.quality_ref_dict = self.get_quality_references()

        # configure
        config_dict = self.get_configure()
        self.config_default = config_dict['default']
        self.data_type_specific_config = config_dict['data_type']

        self.rule_method = self.data_type_specific_config['rule_method']

        # Get the metric type based on the configuration file.
        self.metric_category_names = list(self.config_default[
                                              'category_weights'].keys())  # ['time_domain', 'freq_domain', 'entropy', 'fractal', 'artifacts']
        # cache variances for report
        self.zero_mask = None
        self.nan_mask = None
        self.bad_chan_mask = None
        self.bad_seg_mask = None
        self.flat_mask = None
        self.bad_chan_names = None
        self.bad_chan_index = None

    def get_quality_references(self) -> Dict:
        """
        Load quality reference values based on the MEG data type.

        Returns
        -------
        dict
            A dictionary containing the quality reference values.
        """
        if self.data_type == 'opm':
            quality_ref_fpath = Path(__file__).parent.parent / 'quality_reference' / 'opm_quality_reference.yaml'
        else:
            quality_ref_fpath = Path(__file__).parent.parent / 'quality_reference' / 'squid_quality_reference.yaml'

        quality_ref_dict = read_yaml(quality_ref_fpath)
        return quality_ref_dict

    def get_configure(self) -> Dict:
        """
        Load configuration parameters from configuration file[conf folder].

        Returns
        -------
        dict
            A dictionary containing the configuration parameters.
        """
        conf_path = Path(__file__).parent.parent / "conf"
        default_config = read_yaml(conf_path / "config.yaml")
        specific_config = read_yaml(conf_path / self.data_type / "quality_config.yaml")
        return {"default": default_config, "data_type": specific_config}

    def _get_single_quality_ref_dict(self, metric_name, bound=None, limit=None, method=None) -> Dict:
        """Get single quality reference.

        Parameters
        ----------
        metric_name : str
            the name of metric score
        bound : float, optional
            if the value is not None, recalculating the reference range instead of
            taking the upper and lower bounds in the quality_reference file(*_quality_reference.yaml).
        limit: float, optional
            the value used to calculate upper and lower limits.
        method: str
            IQR method ('iqr') or Sigma method ('sigma')
        Returns
        -------
        dict
            A dictionary containing the reference range values for a single quality metric.
        """
        if method == 'sigma':
            mean = self.quality_ref_dict[metric_name]['mean']
            std = self.quality_ref_dict[metric_name]['std']
            bounds = self.quality_ref_dict[metric_name]['sigma_range']
            if len(bounds) != 2:
                raise ValueError("The length of the quality metric range is incorrect (not equal to 2).")
            lower_bound, upper_bound = bounds[0], bounds[-1]

            maximum_k = mean + limit * std
            minimum_l = mean - limit * std

            if bound is not None:
                lower_bound = mean - bound * std
                upper_bound = mean + bound * std

            # customize limits of artifacts.
            if metric_name in ['BadChanRatio', 'Flat_chan_ratio']:
                maximum_k = self.quality_ref_dict[metric_name]['maximum_k']
                minimum_l = self.quality_ref_dict[metric_name]['minimum_l']

            return {"lower_bound": lower_bound, "upper_bound": upper_bound,
                    "mean:": mean, "std:": std,
                    "maximum_k": maximum_k, "minimum_l": minimum_l}

        elif method == 'iqr':
            q1 = self.quality_ref_dict[metric_name]['q1']
            q3 = self.quality_ref_dict[metric_name]['q3']
            bounds = self.quality_ref_dict[metric_name]['iqr_range']
            if len(bounds) != 2:
                raise ValueError("The length of the quality metric range is incorrect (not equal to 2).")
            lower_bound, upper_bound = bounds[0], bounds[-1]
            maximum_k = q3 + limit * (q3 - q1)
            minimum_l = q1 - limit * (q3 - q1)
            if 'maximum_k' in self.quality_ref_dict[metric_name]:
                maximum_k = self.quality_ref_dict[metric_name]['maximum_k']
            if 'minimum_l' in self.quality_ref_dict[metric_name]:
                minimum_l = self.quality_ref_dict[metric_name]['minimum_l']

            if bound is not None:
                lower_bound = q1 - bound * (q3 - q1)
                upper_bound = q3 + bound * (q3 - q1)

            return {"lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "q1:": q1, "q3": q3,
                    "maximum_k": maximum_k,
                    "minimum_l": minimum_l}
        else:
            return {}

    @staticmethod
    def _calculate_quality_metric(metric_name, raw, meg_type, n_jobs, data_type, origin_raw):
        """
        Calculate quality metrics for a specific domain.

        Parameters
        ----------
        metric_name : str
            Name of the metric domain.
        raw : mne.io.Raw
            The MEG raw data.
        meg_type : MEG_TYPE
            Type of MEG channels ('mag' or 'grad').
        n_jobs : int
            Number of parallel jobs for computation.
        data_type : DATA_TYPE
            Type of MEG data ('opm' or 'squid').
        origin_raw : mne.io.Raw or None
            The original MEG raw data for comparison.

        Returns
        -------
        list
            A list containing the metric DataFrame, cached report, and metric class name.
        """
        try:
            metric_instance = MetricsFactory.create_metric(metric_name, raw=raw, data_type=data_type, n_jobs=n_jobs,
                                                           origin_raw=origin_raw)
            res = metric_instance.compute_metrics(meg_type)
            cache_report = {
                "zero_mask": metric_instance.zero_mask,
                "nan_mask": metric_instance.nan_mask,
                "bad_chan_mask": metric_instance.bad_chan_mask,
                "bad_seg_mask": metric_instance.bad_seg_mask,
                "flat_mask": metric_instance.flat_mask,
                "bad_chan_names": metric_instance.bad_chan_names,
            } if metric_name == "stats_domain" else None
            return [res, cache_report, metric_instance.__class__.__name__]
        except Exception as e:
            clogger.error(f"Error calculating {metric_name}: {e}")
            return [None, None, None]

    def compute_single_metric(self, metric_score, metric_name, method):
        """
        Calculate a single quality metric score based on the reference range.

        Parameters
        ----------
        metric_score : float
            The score of the quality metric.
        metric_name : str
            The name of the metric.
        method : str
            Method for range calculation ('iqr' or 'sigma').

        Returns
        -------
        dict
            A dictionary containing the calculated quality score and related metadata.
        """
        if method == 'sigma':
            bound = self.data_type_specific_config['bound_threshold_std_dev']
            limit = self.data_type_specific_config['limit_threshold_std_dev']
        elif method == 'iqr':
            bound = self.data_type_specific_config['bound_threshold_iqr']
            limit = self.data_type_specific_config['limit_threshold_iqr']
        else:
            bound = None
            limit = None

        single_quality_ref = self._get_single_quality_ref_dict(metric_name, bound=bound,
                                                               limit=limit, method=method)
        lower_bound, upper_bound = single_quality_ref['lower_bound'], single_quality_ref['upper_bound']
        maximum_k, minimum_l = single_quality_ref['maximum_k'], single_quality_ref['minimum_l']

        quality_score = None
        hint = None
        if lower_bound <= metric_score <= upper_bound:
            quality_score = 1
            hint = "✔"
        elif upper_bound < metric_score < maximum_k:
            quality_score = 1 - (metric_score - upper_bound) / (maximum_k - upper_bound)
            hint = "↑"
        elif metric_score <= minimum_l or metric_score >= maximum_k:
            quality_score = 0
            hint = "✘"
        elif minimum_l < metric_score < lower_bound:
            quality_score = 1 - (lower_bound - metric_score) / (lower_bound - minimum_l)
            hint = "↓"

        # check
        if quality_score > 1 or quality_score < 0:
            raise ValueError(f"normative quality score {quality_score} is wrong! Please check your input.")
        return {"quality_score": quality_score,
                "metric_score": metric_score,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "maximum_k": maximum_k,
                "minimum_l": minimum_l,
                "hint": hint}

    def calculate_category_score(self, metrics_df, method):
        """
        Calculate average scores for each metric category.

        Parameters
        ----------
        metrics_df : pd.DataFrame
            DataFrame containing metrics.
        method : str
            Method for score calculation ('iqr' or 'sigma').

        Returns
        -------
        dict
            A dictionary with category scores and detailed metric scores.
        """
        categories_score = {}
        details = {}
        for metric_cate_name in self.metric_category_names:
            metric_column_names = METRICS_COLUMNS[metric_cate_name]
            metrics = metrics_df[metric_column_names].loc['avg_mag'].tolist()
            weights = np.array([1] * len(metrics))
            metrics_score = []

            for idx, metric_name in enumerate(metric_column_names):
                score = self.compute_single_metric(metrics[idx], metric_name, method)
                quality_score = score["quality_score"]
                metrics_score.append(quality_score)
                details[metric_name] = score

            metrics_score = np.array(metrics_score)
            category_score = np.sum(weights * metrics_score) / np.sum(weights)
            categories_score[metric_cate_name] = category_score
        return {"categories_score": categories_score, "details": details}

    def compute_msqm_score(self):
        """
        Compute the MSQM score based on metric categories.

        Returns
        -------
        dict
            The computed MSQM score and detailed scores for each category.

        # For example,
        # compute the msqm score and obtain the reference values & hints[↑↓✔]
        # "msqm_score":98,
        # "S": {"lower_bound","upper_bound,"hint":"✔"}
        # "I": {"score":0.9,"value":10e-12,"lower_bound":,"upper_bound,"hints":"↓"}
        """
        # metric_lists = self._calculate_quality_metric("entropy_domain", self.raw, self.meg_type, self.n_jobs,self.data_type,self.origin_raw) # for fast debug.
        # parallel version.
        # bug for squid: A task has failed to un-serialize. Please ensure that the arguments of the function are all picklable.
        # metric_lists = Parallel(self.n_jobs, verbose=self.verbose)(
        #     delayed(self._calculate_quality_metric)(metric_cate_name, self.raw, self.meg_type, self.n_jobs,
        #                                             self.data_type, self.origin_raw) for metric_cate_name in ["time_domain","freq_domain","entropy_domain","stats_domain"])

        # serial version.
        metric_lists = []
        for metric_cate_name in METRICS_DOMAIN:
            clogger.info(f"Computing metrics for {metric_cate_name}...")
            metric_lists.append(self._calculate_quality_metric(metric_cate_name, raw=self.raw, meg_type=self.meg_type,
                                                               n_jobs=self.n_jobs, data_type=self.data_type,
                                                               origin_raw=self.origin_raw))

        # get metrics and cache mask for reports
        metric_list = []
        cache_report = None
        for i in metric_lists:
            metric_list.append(i[0])
            if i[1] is not None:
                cache_report = i[1]

        if cache_report is not None:
            self.zero_mask = cache_report['zero_mask']
            self.nan_mask = cache_report['nan_mask']
            self.bad_chan_mask = cache_report['bad_chan_mask']
            self.bad_seg_mask = cache_report['bad_seg_mask']
            self.flat_mask = cache_report['flat_mask']
            self.bad_chan_names = cache_report['bad_chan_names']

        metrics_df = pd.concat(metric_list, axis=1)

        category_scores_res = self.calculate_category_score(metrics_df, method=self.rule_method)
        category_scores_dict = category_scores_res['categories_score']
        category_weights_dict = self.config_default['category_weights']

        category_weights = np.array([category_weights_dict[k] for k in self.metric_category_names])
        category_scores = np.array([category_scores_dict[k] for k in self.metric_category_names])

        details = category_scores_res["details"]
        msqm_score = np.sum(category_weights * category_scores) / np.sum(category_weights)

        return {"msqm_score": msqm_score, "details": details, "category_scores": category_scores_dict}
