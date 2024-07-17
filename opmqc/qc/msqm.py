# -*- coding: utf-8 -*-
"""
MEG quality assessment based on MEG Signal Quality Metrics(MSQMs)
"""
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from joblib import Parallel, delayed

from opmqc.qc.metrics import Metrics
from opmqc.constants import DATA_TYPE, METRICS_COLUMNS
from opmqc.utils import read_yaml
from opmqc.qc.freq_domain_metrics import FreqDomainMetrics
from opmqc.qc.tsfresh_domain_metrics import TsfreshDomainMetric
from opmqc.qc.time_domain_metrcis import TimeDomainMetric
from opmqc.qc.statistic_metrics import StatsDomainMetric
from opmqc.qc.entropy_metrics import EntropyDomainMetric
from opmqc.utils import clogger

class MSQM(Metrics):
    def __init__(self, raw: mne.io.Raw, data_type: DATA_TYPE, origin_raw: mne.io.Raw = None, n_jobs=-1, verbose=False):
        """MEG quality assessment based on MEG Signal Quality Metrics (MSQMs)

        Parameters
        ----------
        raw : mne.io.Raw
            the object of the MEG raw data
        origin_raw : mne.io.Raw, optional
            keep the original MEG raw data.
        data_type : DATA_TYPE
            the data type of the MEG data.(opm or squid)
        """
        super().__init__(raw, data_type=data_type, origin_raw=origin_raw, n_jobs=n_jobs, verbose=verbose)
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

    def get_quality_references(self) -> Dict:
        """ Get quality reference according to data type of MEG.(opm or squid)
        """
        if self.data_type == 'opm':
            quality_ref_fpath = Path(__file__).parent.parent / 'quality_reference' / 'opm_quality_reference.yaml'
        else:
            quality_ref_fpath = Path(__file__).parent.parent / 'quality_reference' / 'squid_quality_reference.yaml'
        quality_ref_dict = read_yaml(quality_ref_fpath)

        return quality_ref_dict

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
            Reference range value for a single quality metric score
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
            median = self.quality_ref_dict[metric_name]['median']
            q1 = self.quality_ref_dict[metric_name]['q1']
            q3 = self.quality_ref_dict[metric_name]['q3']
            bounds = self.quality_ref_dict[metric_name]['iqr_range']
            if len(bounds) != 2:
                raise ValueError("The length of the quality metric range is incorrect (not equal to 2).")
            lower_bound, upper_bound = bounds[0], bounds[-1]
            maximum_k = q3 + limit * (q3 - q1)
            minimum_l = q1 - limit * (q3 - q1)

            if bound is not None:
                lower_bound = q1 - bound * (q3 - q1)
                upper_bound = q3 + bound * (q3 - q1)

            # customize limits of artifacts.
            if metric_name in ['BadChanRatio', 'Flat_chan_ratio']:
                maximum_k = self.quality_ref_dict[metric_name]['maximum_k']  # the bad channel limit.
                minimum_l = self.quality_ref_dict[metric_name]['minimum_l']

            return {"lower_bound": lower_bound, "upper_bound": upper_bound,
                    "median:": median, "q1:": q1, "q3": q3,
                    "maximum_k": maximum_k, "minimum_l": minimum_l}
        else:
            return {}

    def get_configure(self) -> Dict:
        """ get configuration parameters from configuration file[conf folder].
        """
        default_config_fpath = Path(__file__).parent.parent / 'conf' / 'config.yaml'
        if self.data_type == 'opm':
            config_fpath = Path(__file__).parent.parent / 'conf' / 'opm' / 'quality_config.yaml'
        else:
            config_fpath = Path(__file__).parent.parent / 'conf' / 'squid' / 'quality_config.yaml'
        config = read_yaml(config_fpath)
        default_config = read_yaml(default_config_fpath)
        return {'default': default_config, 'data_type': config}

    @staticmethod
    def _calculate_quality_metric(metric_name, raw, meg_type, n_jobs, data_type, origin_raw):
        cache_report = None
        if metric_name == "tfresh":
            m = TsfreshDomainMetric(raw, data_type=data_type, n_jobs=n_jobs)
            res = m.compute_tsfresh_metrics(meg_type)
        if metric_name == "time_domain":
            m = TimeDomainMetric(raw, data_type=data_type)
            res = m.compute_time_metrics(meg_type)
        elif metric_name == "freq_domain":
            m = FreqDomainMetrics(raw, data_type=data_type)
            res = m.compute_freq_metrics(meg_type=meg_type)
        elif metric_name == "entropy_domain":
            m = EntropyDomainMetric(raw, n_jobs=n_jobs, data_type=data_type)
            res = m.compute_entropy_metrics(meg_type=meg_type)
        elif metric_name == "stats_domain":
            m = StatsDomainMetric(raw, data_type=data_type, origin_raw=origin_raw)
            res = m.compute_stats_metrics(meg_type)

            # cache for report.
            cache_report = {
                "zero_mask": m.zero_mask,
                "nan_mask": m.nan_mask,
                "bad_chan_mask": m.bad_chan_mask,
                "bad_seg_mask": m.bad_seg_mask,
                "flat_mask": m.flat_mask,
                "bad_chan_names": m.bad_chan_names,
            }

        else:
            res = [None, None]

        return [res, cache_report]

    def compute_single_metric(self, metric_score, metric_name, method):
        """single quality metric score is calculated based on the range of quality metric.

        Parameters
        ----------
        metric_score : float
            the score of quality metric.
        metric_name : str
            the name of the metric.
        method : str
            IQR method ('iqr') or Sigma method ('sigma')
        """
        if method == 'sigma':
            bound = self.data_type_specific_config['bound_threshold_std_dev']
            limit= self.data_type_specific_config['limit_threshold_std_dev']
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
        """average metrics in one category """
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
        for metric_cate_name in ["time_domain", "freq_domain", "stats_domain", "entropy_domain"]:
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
