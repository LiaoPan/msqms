# -*- coding: utf-8 -*-
"""Obtain Basic Info of MEG Data."""

import numpy as np
from box import Box
import os.path as op
from collections import Counter, defaultdict

import mne
from mne.io import read_raw_fif
from mne import channel_type
from mne.utils.misc import _pl
from mne.utils import sizeof_fmt
from mne import pick_types

try:
    from mne.io._digitization import _dig_kind_proper, _dig_kind_rev, _dig_kind_ints
except ImportError:
    # Compatibility for mne==1.6.0
    from mne._fiff._digitization import _dig_kind_proper, _dig_kind_rev, _dig_kind_ints

from msqms.utils import clogger
from msqms.utils import format_timedelta


def get_header_info(raw: mne.io.BaseRaw) -> Box:
    """
    Extract basic information from an MNE Raw object.

    This function processes an MNE Raw object to obtain essential metadata
    about MEG recordings, including participant information,
    channel types, and recording details.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The MNE Raw object containing MEG/EEG data.

    Returns
    -------
    info_box : Box
        A dictionary-like object containing:
        - "basic_info": dict
            General recording metadata, including experimenter, participant,
            digitized points, and file details.
        - "meg_info": dict
            Summary of MEG-specific information such as the number of
            magnetometers, gradiometers, EEG channels, and digitized points.
    """
    assert isinstance(raw, mne.io.BaseRaw)
    info = raw.info

    # Experimenter
    experimenter_info = info.get('experimenter', 'unspecified')

    # Measurement date
    meas_date = info.get('meas_date')
    meas_date_info = meas_date.strftime('%Y-%m-%d %H:%M:%S %Z') if meas_date else 'unspecified'

    # Participant information
    try:
        participant = defaultdict(str, info.get('subject_info', {}))
        sex_dict = {0: 'unknown', 1: 'male', 2: 'female'}
        birthday = '-'.join(map(str, participant.get('birthday', ('unspecified', 'unspecified', 'unspecified'))))
        participant_info = {
            "name": f"{participant['first_name']} {participant['middle_name']} {participant['last_name']}".strip(),
            "birthday": birthday,
            "sex": sex_dict.get(participant['sex'], 'unspecified')
        }
    except Exception as e:
        clogger.error(e)
        participant_info = {"name": "unspecified", "birthday": "unspecified", "sex": "unspecified"}

    # Digitized points
    dig = info.get('dig', [])
    if dig:
        counts = Counter(d['kind'] for d in dig)
        counts = [f"{counts[ii]} {_dig_kind_proper[_dig_kind_rev[ii]]}" for ii in _dig_kind_ints if ii in counts]
        dig_info = f"{len(dig)} item{_pl(len(dig))} ({', '.join(counts)})"
        n_dig = len(dig)
    else:
        dig_info = 'Not available'
        n_dig = 0

    # Channel information
    n_eeg = len(pick_types(info, meg=False, eeg=True))
    n_grad = len(pick_types(info, meg='grad'))
    n_mag = len(pick_types(info, meg='mag'))
    n_stim = len(pick_types(info, stim=True))
    ch_types = [channel_type(info, idx) for idx in range(len(info['chs']))]
    ch_counts = Counter(ch_types)
    chs_info = ', '.join(f"{count} {ch_type.upper()}" for ch_type, count in ch_counts.items())

    # EOG and ECG channels
    eog = ', '.join(np.array(info['ch_names'])[pick_types(info, meg=False, eog=True)]) or '0'
    ecg = ', '.join(np.array(info['ch_names'])[pick_types(info, meg=False, ecg=True)]) or '0'

    # Bad channels
    bad_info = ', '.join(info.get('bads', [])) or 'unspecified'

    # Sampling frequency, lowpass, highpass
    sfreq_info = f"{info['sfreq']:.1f} Hz"
    lowpass_info = f"{info['lowpass']:.1f} Hz"
    highpass_info = f"{info['highpass']:.1f} Hz"

    # Duration
    duration_info = format_timedelta(raw.times[-1]) + " (HH:MM:SS.SSS)"

    # Source filename and size
    source_filename = raw.filenames[0] if raw.filenames else 'unspecified'
    file_size = sizeof_fmt(op.getsize(source_filename)) if source_filename != 'unspecified' else 'unspecified'

    # Prepare output
    basic_info = {
        'Experimenter': experimenter_info,
        'Measurement date': meas_date_info,
        'Participant': participant_info,
        'Digitized points': dig_info,
        'Good channels': chs_info,
        'Bad channels': bad_info,
        'EOG channels': eog,
        'ECG channels': ecg,
        'Sampling frequency': sfreq_info,
        'Highpass': highpass_info,
        'Lowpass': lowpass_info,
        "Duration": duration_info,
        "Source filename": source_filename,
        'Data size': file_size
    }

    meg_info = {
        'n_mag': n_mag,
        'n_grad': n_grad,
        'n_stim': n_stim,
        'n_eeg': n_eeg,
        'n_ecg': ecg,
        'n_eog': eog,
        'n_dig': n_dig
    }

    return Box({"basic_info": basic_info, "meg_info": meg_info})
