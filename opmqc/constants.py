# -*- coding: utf-8 -*-
from typing import TypeVar, Literal

MEG_TYPE = TypeVar("MEG_TYPE", Literal['mag'], Literal['grad'])
DATA_TYPE = TypeVar("DATA_TYPE", Literal['opm'], Literal['squid'])

METRICS_COLUMNS = {
    "time_domain": ['max_ptp', 'S', 'C', 'I', 'L', 'mmr', 'max_field_change', 'mean_field_change', 'std_field_change',
                    'rms', 'arv', 'mean', 'variance', 'std_values', 'max_values', 'min_values', 'median_values',
                    'hjorth_mobility', 'hjorth_complexity', 'num_of_zero_crossings', 'DFA', 'max_mean_offset',
                    'mean_offset', 'Zero_ratio',
                    'std_mean_offset', 'max_median_offset', 'median_offset', 'std_median_offset'],
    "freq_domain": ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13'],
    "fractal": ['PFD', 'KFD', 'HFD'],
    "entropy_domain": ['permutation_entropy', 'spectral_entropy', 'svd_entropy', 'approximate_entropy',
                       'sample_entropy',
                       'power_spectral_entropy', 'Total_Energy', 'Total_Entropy',
                       'Energy_Entropy_Ratio'],
    "artifacts": ['BadChanRatio', 'BadSegmentsNum', 'NaN_ratio', 'Flat_chan_ratio']}
