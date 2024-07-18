# -*- coding: utf-8 -*-
import mne
from opmqc.qc import Metrics
from opmqc.constants import MEG_TYPE
from opmqc.constants import METRICS_DOMAIN,METRICS_COLUMNS

class MetricsFactory:
    _registry = {}
    _add_domain = None
    @classmethod
    def register_metric(cls, name, metric_class):
        """Used to register new metric classes"""
        if not issubclass(metric_class, Metrics):
            raise ValueError(f"{metric_class} must be a subclass of Metric")
        cls._registry[name] = metric_class

    @classmethod
    def create_metric(cls, name, *args, **kwargs):
        """Create a metric class instance by name"""
        metric_class = cls._registry.get(name)
        if not metric_class:
            raise ValueError(f"No metric registered under name: {name}")
        return metric_class(*args, **kwargs)

    @classmethod
    def register_custom_metric(cls, name, func, custom_metrics_name):
        METRICS_DOMAIN.append("custom_domain")
        METRICS_COLUMNS[name].extend(custom_metrics_name)
        cls._add_domain = name
        class CustomMetric(Metrics):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.args = args
                self.kwargs = kwargs

            def compute_metrics(self, meg_type: MEG_TYPE):
                meg_metrics_df = func(self, meg_type=meg_type)
                meg_metrics_df.loc[f"avg_{meg_type}"] = meg_metrics_df.mean(axis=0)
                meg_metrics_df.loc[f"std_{meg_type}"] = meg_metrics_df.std(axis=0)
                return meg_metrics_df

        cls.register_metric("custom_domain", CustomMetric)
