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



def segment_raw_data(raw, seg_length: float):
    """将Raw（mne.io.Raw）数据分段，方便拆分计算。
    seg_length:表示分割的长度，以时间秒来算。
    """
    raw_list = []
    first_time = raw.first_time
    last_time = raw._last_time
    duration = last_time - first_time
    segment_times = []
    for i in np.arange(0,duration, seg_length):
        if i+seg_length <= duration:
            segment_times.append([i, i+seg_length])
            raw_list.append(raw.copy().crop(i, i+seg_length))
        else:
            segment_times.append([i, duration])
            raw_list.append(raw.copy().crop(i, duration))
    print(segment_times)

    return raw_list,segment_times