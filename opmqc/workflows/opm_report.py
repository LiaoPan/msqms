# -*- coding: utf-8 -*-
"""OPM-MEG Quality Report Generation Pipeline"""

from opmqc.qc import qc_metrics
from opmqc.io import read_raw_mag
from mne.io import read_raw_fif

from opmqc.main import test_opm_fif_path, test_opm_mag_path, test_squid_fif_path
from opmqc.reports import gen_quality_report
from opmqc.qc import get_header_info

raw = read_raw_fif(test_opm_fif_path, verbose=False)
info = get_header_info(raw)
print("Basic Info:", info)

gen_quality_report()