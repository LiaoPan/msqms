# -*- coding: utf-8 -*-
"""Utility functions"""
import yaml
import mne.io
import datetime
import numpy as np
from typing import Dict, Any
from pathlib import Path
from mne.io import RawArray
from msqms.constants import DATA_TYPE


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


def save_yaml(data, fname_path):
    """
    Save a dictionary as a YAML file.

    Parameters
    ----------
    data : dict
        The data to be saved in YAML format.
    fname_path : str or Path
        The path where the YAML file will be saved.

    Returns
    -------
    None

    Notes
    -----
    This function will overwrite the file if it already exists.
    """
    with open(fname_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

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


def get_configure(data_type: DATA_TYPE) -> Dict:
    """get configuration parameters from configuration file[conf folder].

    Parameters
    ----------
    data_type : DATA_TYPE
        the data type of MEG.('opm' or 'squid')
    Returns
    -------
        the dict of configuration parameters,including 'default' and 'data_type'.
    """
    default_config_fpath = Path(__file__).parent.parent / 'conf' / 'config.yaml'
    if data_type == 'opm':
        config_fpath = Path(__file__).parent.parent / 'conf' / 'opm' / 'quality_config.yaml'
    elif data_type == 'squid':
        config_fpath = Path(__file__).parent.parent / 'conf' / 'squid' / 'quality_config.yaml'
    else:
        raise ValueError(f'{data_type} is not a valid')
    config = read_yaml(config_fpath)
    default_config = read_yaml(default_config_fpath)
    return {'default': default_config, 'data_type': config}


def normative_score(num, thres=20):
    """normative score.
    """
    return 1 - 1 / (1 + (num / thres) ** 2)


def check_if_directory(path: Path):
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(f"The path '{path}' is not a directory.")
    else:
        print(f"The path '{path}' is a valid directory.")

def filter(raw: mne.io.Raw, high_pass: float, low_pass: float, notch_freq: [float], data_type: DATA_TYPE, pad_length=10,
           n_jobs=-1, verbose=True) -> RawArray | Any:
    """Filter in different ways for different data types

    Parameters
    ----------
    raw : mne.io.Raw
    data_type : Data_TYPE
        the data type of MEG.('opm' or 'squid')
    high_pass: float
        the high pass frequency.
    low_pass: float
        the low pass frequency.
    notch_freq: list[float]
        the notch filter frequency.
    pad_length: int
        the padding of data before filtering (seconds),`reflect` fill before and after the data.
    n_jobs: int
        the number of jobs.
    Returns
    -------
        the filtered raw
    """
    raw_filter = raw.copy()
    if data_type == 'opm':
        # Mitigate the edge effect of opm.(but there will still be residue.)
        raw_filter.notch_filter(notch_freq, n_jobs=n_jobs, verbose=verbose)

        data = raw_filter.get_data()
        sfreq = raw_filter.info['sfreq']

        # obtain the length of pad.
        pad_length = int(sfreq * pad_length)
        # reflect fill before and after the data.
        padded_data = np.pad(data, ((0, 0), (pad_length, pad_length)), mode='reflect')

        # filter
        filtered_padded_data = mne.filter.filter_data(data=padded_data, sfreq=sfreq, l_freq=high_pass, h_freq=low_pass,
                                                      n_jobs=n_jobs, verbose=verbose)

        # remove padding
        filtered_data = filtered_padded_data[:, pad_length:-pad_length]

        # update raw & info
        raw_filter._data = filtered_data
        raw_filter.info._unlocked = True
        raw_filter.info['highpass'] = high_pass
        raw_filter.info['lowpass'] = low_pass
        raw_filter.info._unlocked = False

    else:
        raw_filter.notch_filter(notch_freq, n_jobs=n_jobs, verbose=verbose).filter(l_freq=high_pass, h_freq=low_pass,
                                                                                   n_jobs=n_jobs, verbose=verbose)
    return raw_filter
