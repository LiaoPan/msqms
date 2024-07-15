# coding:utf-8
"""
For debug purposes only
"""
import platform
import hydra
from omegaconf import DictConfig, OmegaConf
import tempfile
import warnings

warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)

system_platform = platform.system()

if system_platform == "mac":
    test_opm_mag_path = "/Volumes/Touch/Datasets/OPM_Dataset/CMR_OPM_HuaRou/xuwei/Artifact/S01.LP.mag"
    test_opm_fif_path = "/Users/reallo/Downloads/opm_artifacts/ta80_raw.fif"
    test_squid_fif_path = "/Volumes/Touch/Datasets/MEG_Lab/02_liaopan/231123/run1_tsss.fif"
else:
    test_opm_mag_path = r"C:\Data\Datasets\Artifact\S01.LP.mag"
    test_opm_fif_path = r"C:\Data\Datasets\OPM-Artifacts\S01.LP.fif"
    # test_opm_fif_path = r"C:\Data\Code\opmqc\sub-01.fif"
    test_squid_fif_path = r"C:\Data\Datasets\MEG_Lab/02_liaopan/231123/run1_tsss.fif"
    opm_visual_fif_path = r"C:\Data\Datasets\全记录数据\opm_visual.fif"

@hydra.main(config_path='conf', config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point

    Parameters
    ----------
    cfg : DictConfig
        configuration file

    Returns
    -------
        None
    """
    print(OmegaConf.to_yaml(cfg))
    update_config_yaml(cfg,conf_path="temp.yaml") # will auto save into outputs/{timestamp}/temp.yaml
     
def update_config_yaml(conf:DictConfig,conf_path:str):
    """save configure yaml file.

    Parameters:
        conf : DictConfig
            config object(omegaconf.DictConfig)
        conf_path : str
            file path with *.yaml

    Examples:
        update_config_yaml(OmegaConf.create({"opm":123,"squid":123}),"./conf/temp.yaml")
    """
    assert isinstance(conf,DictConfig),"conf must be DictConfig"
    OmegaConf.save(config=conf,f=conf_path)
    print("saving f.name:",conf_path)



