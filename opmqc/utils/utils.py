# -*- coding: utf-8 -*-
"""Utility functions"""

import datetime
import numpy as np
import yaml
from typing import Dict
from pathlib import Path

def fill_zeros_with_nearest_value(arr):
    """find zeros value, interpolate arr with nearest value."""
    zero_indices = np.where(arr == 0)[0]
    non_zero_indices = np.where(arr != 0)[0]
    for idx in zero_indices:
        left_idx = non_zero_indices[non_zero_indices < idx][-1] if np.any(non_zero_indices < idx) else -np.inf
        right_idx = non_zero_indices[non_zero_indices > idx][0] if np.any(non_zero_indices > idx) else np.inf
        arr[idx] = arr[left_idx] if (idx - left_idx) < (right_idx - idx) else arr[right_idx]
    return arr


def format_timedelta(seconds):
    """convert seconds to HH:MM:SS+MS"""
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = "{:02}:{:02}:{:06.3f}".format(hours, minutes, seconds + delta.microseconds / 1e6)
    return formatted_time


def segment_raw_data(raw, seg_length: float):
    """The Raw (mne.io.Raw) data is segmented to facilitate metrics calculation.

    Parameters
    ----------
    raw : mne.io.raw
        the object of MEG data.
    seg_length : float
        Represents the length of the split (seconds).

    Returns
    -------
        raw_list : [mne.io.raw]
            the list of segmented raw.
        segment_times : list
            the list of segmented times.
    """
    raw_list = []
    first_time = raw.first_time
    last_time = raw._last_time
    duration = last_time - first_time
    segment_times = []
    for i in np.arange(0, duration, seg_length):
        if i + seg_length <= duration:
            segment_times.append([i, i + seg_length])
            raw_list.append(raw.copy().crop(i, i + seg_length))
        else:
            segment_times.append([i, duration])
            raw_list.append(raw.copy().crop(i, duration))
    return raw_list, segment_times


def read_yaml(yaml_file):
    """Read yaml file

    Parameters
    ----------
    yaml_file : str | Path
        the path of the yaml file.
    Returns
    -------
    content : dict
        the contents of the yaml file.
    """
    with open(yaml_file, 'r') as file:
        content = yaml.safe_load(file)
    return content

def normative_score(num,thres=20):
    """normative score.
    """
    return 1 - 1/(1+(num/thres)**2)