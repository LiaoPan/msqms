# -*- coding: utf-8 -*-
"""temporally record all quality control metric."""

import mne
import numpy as np
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew,kurtosis
class QC_Metrics:
    def __init__(self,raw:mne.io.Raw):
        self.raw = raw
        self.bad_channels = []

    def tsfresh_metrics(self):
        pass

    def entropy_metrics(self):
        pass

    def freq_metrics(self):
        pass

    def time_domain_metrics(self):
        pass

    def stats_domain_metrics(self):
        pass




if __name__ == '__main__':
   from opmqc.main import opm_visual_fif_path
   import mne
   from pathlib import Path

   opm_mag_fif = "C:\Data\Datasets\Artifact\S01.LP.fif"
   opm_raw = mne.io.read_raw(opm_mag_fif, verbose=False)
   opm_raw.first_samp

