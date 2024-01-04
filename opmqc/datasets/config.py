# -*- coding: utf-8 -*-
"""Constant Configs"""

# Download url of MEG datasets(e.g. OPM or SQUID)
MEG_DATASETS = dict()

# Private Local OPM datasets
MEG_DATASETS["local_opm_cmr_a"] = dict(
    archive_name="opm_cmr",
    url="/Volumes/Touch/Datasets/OPM-DATA/raw/S01/A/",
    file_list=["raw-run1.mat",
               "raw-run2.mat",
               "raw-run3.mat"]
)

MEG_DATASETS["local_opm_cmr_b"] = dict(
    archive_name="opm_cmr",
    url="/Volumes/Touch/Datasets/OPM-DATA/raw/S01/B/",
    file_list=["raw-run1.mat",
               "raw-run2.mat",
               "raw-run3.mat"]
)



MEG_DATASETS["local_opm_mne"] = dict(
    archive_name="opm_mne",
    url="/Volumes/Touch/Datasets/OPM_Dataset/MNE-OPM-data/MEG/OPM/",
    file_list=["OPM_empty_room.fif",
               "OPM_resting_state_raw.fif",
               "OPM_SEF_raw.fif"]
)

MEG_DATASETS["local_squid_mne"] = dict(
    archive_name="squid_mne",
    url="/Volumes/Touch/Datasets/OPM_Dataset/MNE-OPM-data/MEG/SQUID/",
    file_list=["SQUID_empty_room.fif",
               "SQUID_resting_state_raw.fif"
               "SQUID_SEF.fif"]
)

# Public OPM datasets
MEG_DATASETS["opm_cmr"] = dict(
    archive_name="cmr_opm-data.tar.gz",
    hash="",
    url="https://",
    config_key="CMR_DATASETS_OPM_PATH"
)  # noqa

MEG_DATASETS["ucl_opm_auditory"] = dict(
    archive_name="auditory_OPM_stationary.zip",
    hash="md5:9ed0d8d554894542b56f8e7c4c0041fe",
    url="https://osf.io/download/mwrt3/?version=1",
    folder_name="auditory_OPM_stationary",
    config_key="MNE_DATASETS_UCL_OPM_AUDITORY_PATH",
)

MEG_DATASETS["opm"] = dict(
    archive_name="MNE-OPM-data.tar.gz",
    hash="md5:370ad1dcfd5c47e029e692c85358a374",
    url="https://osf.io/p6ae7/download?version=2",
    folder_name="MNE-OPM-data",
    config_key="MNE_DATASETS_OPM_PATH",
)

# Public SQUID datasets
MEG_DATASETS["sample"] = dict(
    archive_name="MNE-sample-data-processed.tar.gz",
    hash="md5:e8f30c4516abdc12a0c08e6bae57409c",
    url="https://osf.io/86qa2/download?version=6",
    folder_name="MNE-sample-data",
    config_key="MNE_DATASETS_SAMPLE_PATH",
)
