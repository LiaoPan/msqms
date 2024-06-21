# -*- coding: utf-8 -*-
"""Obtain Basic Info of MEG Data."""

import numpy as np
import os.path as op

import mne
from mne.io import read_raw_fif
from mne import channel_type
from mne.utils.misc import _pl
from mne.utils import sizeof_fmt
from mne import pick_types
from opmqc.utils import clogger

try:
    from mne.io._digitization import _dig_kind_proper, _dig_kind_rev, _dig_kind_ints
except ImportError:
    # for mne==1.6.0
    from mne._fiff._digitization import _dig_kind_proper, _dig_kind_rev, _dig_kind_ints

from collections import Counter, defaultdict
from box import Box

from opmqc.io import read_raw_mag
from opmqc.utils import format_timedelta


def get_header_info(raw):
    """
    get basic info from MNE.Raw object.
    Parameters
    ----------
    raw

    Returns
    -------
      basic_info: dict
      meg_info : dict
    """
    print("debug---",raw)
    assert isinstance(raw, mne.io.BaseRaw)
    basic_info = {'Experimenter': None, 'Measurement date': None, 'Participant': '', 'Digitized points': None,
                  'Good channels': None, 'Bad channels': None, 'EOG channels': None, 'ECG channels': None,
                  'Sampling frequency': None, 'Highpass': None, 'Lowpass': None,
                  "Duration": None, "Source filename": None, 'Data Size': None}

    info = raw.info

    # Experimenter
    experimenter_info = info['experimenter']

    # Measurement date
    meas_date = info['meas_date']
    if meas_date is None:
        meas_date_info = 'unspecified'
    else:
        meas_date_info = meas_date.strftime('%Y-%m-%d %H:%M:%S %Z')

    # Participant
    try:
        participant = defaultdict(str, info['subject_info'])
        sex_dict = defaultdict(str, {0: 'unknown', 1: 'male', 2: 'female'})
        if participant['birthday']:
            birthday = '{0}-{1}-{2}'.format(participant['birthday'][0], participant['birthday'][1],
                                            participant['birthday'][2])
        else:
            birthday = 'unspecified'

        participant_info = {"name": participant['first_name'] + participant['middle_name'] + participant['last_name'],
                            "birthday": birthday,
                            "sex": sex_dict[participant['sex']]}
    except Exception as e:
        clogger.error(e)
        participant_info = {"name": "", "birthday": "", "sex": ""}

    # Digitized points
    dig = info['dig']
    if dig is not None:
        counts = Counter(d['kind'] for d in dig)
        counts = ['%d %s' % (counts[ii],
                             _dig_kind_proper[_dig_kind_rev[ii]])
                  for ii in _dig_kind_ints if ii in counts]
        counts = (' (%s)' % (', '.join(counts))) if len(counts) else ''
        dig_info = '%d item%s%s' % (len(dig), _pl(len(dig)), counts)

        # simple ver.
        n_dig = len(info['dig'])
        # dig_info = f"{n_dig} points"
    else:
        dig_info = 'Not available'
        n_dig = 0

    # Good channels
    n_eeg = len(pick_types(raw.info, meg=False, eeg=True))
    n_grad = len(pick_types(raw.info, meg='grad'))
    n_mag = len(pick_types(raw.info, meg='mag'))
    n_stim = len(pick_types(raw.info, stim=True))
    good_ch_info = f'{n_mag} magnetometer, {n_grad} gradiometer, and {n_eeg} EEG channels'

    ch_types = [channel_type(info, idx) for idx in range(len(info['chs']))]
    ch_counts = Counter(ch_types)
    chs_info = "%s" % ', '.join("%d %s" % (count, ch_type.upper())
                                for ch_type, count in ch_counts.items())

    # ECG & EOG
    pick_eog = pick_types(raw.info, meg=False, eog=True)
    if len(pick_eog) > 0:
        eog = ', '.join(np.array(raw.info['ch_names'])[pick_eog])
    else:
        eog = '0'  # 'Not available'
    pick_ecg = pick_types(raw.info, meg=False, ecg=True)
    if len(pick_ecg) > 0:
        ecg = ', '.join(np.array(raw.info['ch_names'])[pick_ecg])
    else:
        ecg = '0'

    # Bad channels
    if info['bads'] is not None:
        bad_info = ', '.join(info['bads'])
    else:
        bad_info = 'unspecified'

    # Sampling frequency & LowPass & HighPass
    sfreq_info = '{:.1f} Hz'.format(info['sfreq'])
    lowpass_info = '{:.1f} Hz'.format(info['lowpass'])
    highpass_info = '{:.1f} Hz'.format(info['highpass'])

    # Duration
    duration_info = format_timedelta(raw.times[-1]) + ' (HH:MM:SS.SSS)'

    # Source filename
    source_filename = raw.filenames[0]

    # Data size
    file_size = sizeof_fmt(op.getsize(source_filename))

    basic_info = {'Experimenter': experimenter_info, 'Measurement date': meas_date_info,
                  'Participant': participant_info,
                  'Digitized points': dig_info,
                  'Good channels': chs_info, 'Bad channels': bad_info, 'EOG channels': eog, 'ECG channels': ecg,
                  'Sampling frequency': sfreq_info, 'Highpass': highpass_info, 'Lowpass': lowpass_info,
                  "Duration": duration_info, "Source filename": source_filename, 'Data size': file_size}

    meg_info = {'n_mag': n_mag, 'n_grad': n_grad, 'n_stim': n_stim, 'n_eeg': n_eeg, 'n_ecg': ecg, 'n_eog': eog,
                'n_dig': n_dig}

    return Box({"basic_info": basic_info, "meg_info": meg_info})


if __name__ == "__main__":
    from opmqc.main import test_opm_fif_path, test_squid_fif_path

    # raw = read_raw_fif(test_opm_fif_path,verbose=False)
    raw = read_raw_fif(test_squid_fif_path, verbose=False)
    print(raw.info)
    info = get_header_info(raw)
    print(info)
