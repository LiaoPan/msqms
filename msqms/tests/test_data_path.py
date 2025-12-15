# coding:utf-8
"""
For debug purposes only
"""
import platform
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

system_platform = platform.system()

if system_platform == "mac":
    test_opm_mag_path = "/Volumes/Touch/Datasets/OPM_Dataset/CMR_OPM_HuaRou/xuwei/Artifact/S01.LP.mag"
    test_opm_fif_path = "/Users/reallo/Downloads/opm_artifacts/ta80_raw.fif"
    test_squid_fif_path = "/Volumes/Touch/Datasets/MEG_Lab/02_liaopan/231123/run1_tsss.fif"
else:
    test_opm_mag_path = r"C:\Data\Datasets\Artifact\S01.LP.mag"
    test_opm_fif_path = r"C:\Data\Datasets\OPM-Artifacts\S01.LP.fif"
    test_squid_fif_path = r"C:\Data\Datasets\MEG_Lab/02_liaopan/231123/run1_tsss.fif"
    opm_visual_fif_path = r"C:\Data\Datasets\全记录数据\opm_visual.fif"
