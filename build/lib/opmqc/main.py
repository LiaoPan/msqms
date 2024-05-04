# coding:utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
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
    test_opm_mag_path = "C:\Data\Datasets\Artifact\S01.LP.mag"
    test_opm_fif_path = "C:\Data\Datasets\opm_artifacts\ta80_raw.fif"
    test_squid_fif_path = "C:\Data\Datasets\MEG_Lab/02_liaopan/231123/run1_tsss.fif"


@hydra.main(config_path='conf',config_name="config")
def main(cfg:DictConfig)->None:
    print(OmegaConf.to_yaml(cfg))
    update_config_yaml(cfg,conf_path="temp.yaml") # will auto save into outputs/{timestamp}/temp.yaml
     
def update_config_yaml(conf:DictConfig,conf_path:str):
    """save configure yaml file.
    Args:
        conf (DictConfig): config object(omegaconf.DictConfig)
        conf_path (str):  file path with *.yaml
    Example:
        update_config_yaml(OmegaConf.create({"opm":123,"squid":123}),"./conf/temp.yaml")
    """
    assert isinstance(conf,DictConfig),"conf must be DictConfig"
    OmegaConf.save(config=conf,f=conf_path)
    print("saving f.name:",conf_path)
 


class TtClass(object):
    """
    It is Test Class, Response for Test;It is Test Class, Response for Test;It is Test Class, Response for Test;It is Test Class, Response for Test
    """

    def Ttdemo(self, cont: str):
        """
        It is a demo of TtClass.
        It is a demo of TtClass.
        It is a demo of TtClass.
        :param cont: str for count.
        :type cont: str
        :return:
        """
        print("Test sphinx auto generate documents.")


def print_hi(name: str, age: int):
    """blah blah blah

    :param name: string somthing
    :type name: str

    :param age: integer
    :type age: int

    :return:
    """
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}-{str(age)}')  # Press ⌘F8 to toggle the breakpoint.


def func(path, field_storage, temporary):
    """基本描述 Numpy Style

    Parameters
    ----------
    path : str
        The path of the file to wrapssss
    field_storage : FileStorage
        The :class:`FileStorage` instance to wrap
    temporary : bool
        Whether or not to delete the file when the File instance is destructed

    Returns
    -------
    BufferedFileStorage
        A buffered writable file descriptor
    """
    print("somth")


class DemoClass(object):
    """
    Demo Project
    """

    def defa(self, s: str, b: int) -> str:
        """

        Parameters
        ----------
        s:str

        b

        Returns
        -------

        """
        pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Hello,OPMQC', 12)
    main()
    # update_config_yaml(OmegaConf.create({"opm":123,"squid":123}),"./conf/temp2.yaml")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
