# -*- coding: utf-8 -*-
"""
Factory class for creating and managing different metrics for MEG data.
"""

from msqms.qc import Metrics
from msqms.constants import MEG_TYPE, METRICS_DOMAIN, METRICS_COLUMNS


class MetricsFactory:
    """
    A factory class for creating and managing metrics for MEG data quality control.

    This class supports registering new metric classes, creating metric instances,
    and registering custom metric implementations.

    Attributes
    ----------
    _registry : dict
        A registry that maps metric names to their corresponding classes.
    _add_domain : str or None
        The name of the most recently added custom domain.

    Methods
    -------
    register_metric(name, metric_class)
        Registers a new metric class by name.
    create_metric(name, *args, **kwargs)
        Creates an instance of a registered metric class.
    register_custom_metric(name, func, custom_metrics_name)
        Registers a custom metric class using a user-defined function.
    """
    _registry = {}
    _add_domain = None

    @classmethod
    def register_metric(cls, name: str, metric_class: type):
        """
        Register a new metric class.

        Parameters
        ----------
        name : str
            The name of the metric to register.
        metric_class : type
            The metric class to register. Must be a subclass of `Metrics`.

        Raises
        ------
        ValueError
            If the provided class is not a subclass of `Metrics`.
        """
        if not issubclass(metric_class, Metrics):
            raise ValueError(f"{metric_class} must be a subclass of Metrics")
        cls._registry[name] = metric_class

    @classmethod
    def create_metric(cls, name: str, *args, **kwargs):
        """
        Create an instance of a registered metric class.

        Parameters
        ----------
        name : str
            The name of the registered metric to instantiate.
        *args
            Positional arguments to pass to the metric class constructor.
        **kwargs
            Keyword arguments to pass to the metric class constructor.

        Returns
        -------
        instance : Metrics
            An instance of the specified metric class.

        Raises
        ------
        ValueError
            If no metric class is registered under the given name.
        """
        metric_class = cls._registry.get(name)
        if not metric_class:
            raise ValueError(f"No metric registered under name: {name}")
        return metric_class(*args, **kwargs)

    @classmethod
    def register_custom_metric(cls, name: str, func, custom_metrics_name: list):
        """
        Register a custom metric class using a user-defined function.

        Parameters
        ----------
        name : str
            The name of the custom metric domain.
        func : callable
            A function that defines how to compute the custom metrics. The function
            should accept a `Metrics` instance as its first argument and `meg_type` as a keyword argument.
        custom_metrics_name : list of str
            Names of the custom metrics computed by the function.

        Notes
        -----
        - The new custom domain is added to `METRICS_DOMAIN`.
        - The corresponding metric names are appended to `METRICS_COLUMNS`.
        """
        METRICS_DOMAIN.append("custom_domain")
        METRICS_COLUMNS[name].extend(custom_metrics_name)
        cls._add_domain = name

        class CustomMetric(Metrics):
            """
            A custom metric class for user-defined metrics.

            Parameters
            ----------
            *args
                Positional arguments for the metric class.
            **kwargs
                Keyword arguments for the metric class.
            """
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.args = args
                self.kwargs = kwargs

            def compute_metrics(self, meg_type: MEG_TYPE):
                """
                Compute metrics for the custom domain.

                Parameters
                ----------
                meg_type : MEG_TYPE
                    Type of MEG channels to process ('mag' or 'grad').

                Returns
                -------
                meg_metrics_df : pd.DataFrame
                    A DataFrame containing the custom metrics with average and standard deviation added.
                """
                meg_metrics_df = func(self, meg_type=meg_type)
                meg_metrics_df.loc[f"avg_{meg_type}"] = meg_metrics_df.mean(axis=0)
                meg_metrics_df.loc[f"std_{meg_type}"] = meg_metrics_df.std(axis=0)
                return meg_metrics_df

        cls.register_metric("custom_domain", CustomMetric)