# -*- coding: utf-8 -*-
"""Utility functions"""

import datetime
import numpy as np

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