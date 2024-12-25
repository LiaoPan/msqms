# -*- coding: utf-8 -*-
from typing import TypeVar, Literal

MEG_TYPE = TypeVar("MEG_TYPE", Literal['mag'], Literal['grad'])
DATA_TYPE = TypeVar("DATA_TYPE", Literal['opm'], Literal['squid'])

# METRICS_CLASS_MAPPING = {"FreqDomainMetric": "frequency_domain",
#                          "StatsDomainMetric": "stats_domain",
#                          "EntropyDomainMetric": "entropy",
#                          "TimeDomainMetric": "time_domain",
#                          "CustomMetric": "custom_domain"}




# For HTML Report Display
## Report Metrics
METRICS_COLUMNS = {
    "time_domain": ['form_factor', 'peak_factor', 'pulse_factor', 'hjorth_mobility', 'hjorth_complexity', 'DFA'],
    "frequency_domain": ['skewness_amplitude', 'kurtosis_amplitude', 'mean_frequency', 'rms_frequency',
                         'fourth_moment_frequency', 'normalized_second_moment', 'frequency_skewness', 'frequency_kurtosis'],
    "fractal": ['PFD', 'KFD', 'HFD'],
    "entropy": ['permutation_entropy', 'spectral_entropy', 'svd_entropy', 'Total_Entropy'],
    "artifacts": ['BadChanRatio', 'BadSegmentsRatio', 'NaN_ratio', 'Flat_chan_ratio']}


## For metric category mappings
METRICS_REPORT_MAPPING = {"time_domain": "Time Metrics",
                          "frequency_domain": "Frequency Metrics",
                          "entropy": "Entropy Metrics",
                          "fractal": "Fractal Metrics",
                          "artifacts": "Artifacts"
                          }

METRICS_DOMAIN = ["time_domain", "freq_domain", "stats_domain", "entropy_domain", "fractal_domain"]

## For single metric mappings
METRICS_MAPPING = {
    "NaN_ratio": "Ratio of No-signal",
    "Flat_chan_ratio": "Ratio of FlatChannels",
    "BadChanRatio": "Ratio of BadChannels",
    "BadSegmentsRatio": "Ratio of BadSegments"
}
