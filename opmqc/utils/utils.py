# -*- coding: utf-8 -*-
"""Utility functions"""
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