# -*- coding: utf-8 -*-
from typing import TypeVar, Literal

MEG_TYPE = TypeVar("MEG_TYPE", Literal['mag'], Literal['grad'])
DATA_TYPE = TypeVar("DATA_TYPE", Literal['opm'], Literal['squid'])

METRICS_COLUMNS = {
    "time_domain": ['max_ptp', 'S', 'C', 'I', 'L', 'mmr', 'max_field_change', 'mean_field_change', 'std_field_change',
                    'rms', 'arv',  'variance', 'std_values', 'max_values', 'min_values',
                    'hjorth_mobility', 'hjorth_complexity',  'DFA'],
    "frequency_domain": ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13'],
    "fractal": ['PFD', 'KFD', 'HFD'],
    "entropy": ['permutation_entropy', 'spectral_entropy', 'svd_entropy', 'approximate_entropy',
                'sample_entropy', 'power_spectral_entropy', 'Total_Energy', 'Total_Entropy', 'Energy_Entropy_Ratio'],
    "artifacts": ['BadChanRatio', 'BadSegmentsRatio', 'NaN_ratio', 'Flat_chan_ratio']}

# For HTML Report Display
## For metric category mappings
METRICS_REPORT_MAPPING = {"time_domain": "Time Metrics",
                          "frequency_domain": "Frequency Metrics",
                          "entropy": "Entropy Metrics",
                          "fractal": "Fractal Metrics",
                          "artifacts": "Artifacts",
                          }

## For single metric mappings
METRICS_MAPPING = {
    "NaN_ratio": "Ratio of No-signal",
    "Flat_chan_ratio": "Ratio of FlatChannels",
    "BadChanRatio": "Ratio of BadChannels",
    "BadSegmentsRatio": "Ratio of BadSegments"
}

