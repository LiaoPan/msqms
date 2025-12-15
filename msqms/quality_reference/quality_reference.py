# -*- coding: utf-8 -*-
"""
Module for calculating the quality reference range for given datasets, including BIDS or RAW datasets.
"""

import os
import yaml
import click
import mne
import pandas as pd
from pathlib import Path
from typing import Union
from msqms.utils import clogger
from msqms.io import read_raw_dataset, read_raw_bids_dataset
from msqms.constants import METRICS_DOMAIN, DATA_TYPE
from msqms.qc.time_domain_metrcis import TimeDomainMetric
from msqms.qc.freq_domain_metrics import FreqDomainMetric
from msqms.qc.statistic_metrics import StatsDomainMetric
from msqms.qc.entropy_metrics import EntropyDomainMetric
from msqms.qc.fractal_metrics import FractalDomainMetric
from msqms.qc.metrics_factory import MetricsFactory

# Register metrics classes based on METRICS_DOMAIN
for domain in METRICS_DOMAIN:
    METRICS_CLASS_MAPPING = {
        "time_domain": TimeDomainMetric,
        "freq_domain": FreqDomainMetric,
        "stats_domain": StatsDomainMetric,
        "entropy_domain": EntropyDomainMetric,
        "fractal_domain": FractalDomainMetric
    }

    metric_class = METRICS_CLASS_MAPPING.get(domain)
    if metric_class:
        MetricsFactory.register_metric(domain, metric_class)


def calculate_quality_metrics(meg_filename, data_type=DATA_TYPE, n_jobs=-1):
    """
    Calculate quality metrics for the given MEG file.

    Parameters
    ----------
    meg_filename : str
        Path to the MEG file.
    data_type : DATA_TYPE
        The data type, either "mag"
    n_jobs : int
        Number of parallel jobs for computation.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the computed quality metrics.
    """
    try:
        origin_raw = mne.io.read_raw(meg_filename, verbose=False, preload=True)
        raw = origin_raw.copy().filter(1, 100, n_jobs=n_jobs, verbose=False).notch_filter([50, 100], verbose=False,
                                                                                          n_jobs=n_jobs)
    except ValueError as e:
        clogger.error(f"Error reading MEG file: {e}")
        return None
    metric_list = []
    for metric_cate_name in METRICS_DOMAIN:
        metric_instance = MetricsFactory.create_metric(metric_cate_name, raw=raw, data_type=data_type, n_jobs=n_jobs,
                                                       origin_raw=origin_raw)
        res = metric_instance.compute_metrics(meg_type='mag')
        metric_list.append(res)

    metrics_df = pd.concat(metric_list, axis=1)
    return metrics_df


def obtain_quality_ranges(dataset_metrics_df, sigma=1, k=0.5):
    """
    Calculate quality bounds (sigma and IQR) based on the dataset metrics.

    Parameters
    ----------
    dataset_metrics_df : pd.DataFrame
        DataFrame containing the quality metrics.
    sigma : float, optional
        Number of standard deviations for sigma bounds. Default is 1.
    k : float, optional
        Scaling factor for IQR bounds. Default is 0.5.

    Returns
    -------
    pd.DataFrame
        The updated dataset metrics DataFrame with calculated quality bounds.
    """
    avg_mag = dataset_metrics_df.loc['avg']
    std_mag = dataset_metrics_df.loc['std']
    q3_mag = dataset_metrics_df.loc['q3']
    q1_mag = dataset_metrics_df.loc['q1']
    iqr_mag = q3_mag - q1_mag
    # median_mag = dataset_metrics_df.loc['median']

    sigma_upper_bound = avg_mag + sigma * std_mag
    sigma_lower_bound = avg_mag - sigma * std_mag
    iqr_upper_bound = q3_mag + k * iqr_mag
    iqr_lower_bound = q1_mag - k * iqr_mag

    dataset_metrics_df.loc['sigma_upper_bound'] = sigma_upper_bound
    dataset_metrics_df.loc['sigma_lower_bound'] = sigma_lower_bound
    dataset_metrics_df.loc['iqr_upper_bound'] = iqr_upper_bound
    dataset_metrics_df.loc['iqr_lower_bound'] = iqr_lower_bound

    return dataset_metrics_df


def convert_to_yaml(quality_bounds_df):
    """
    Convert the quality bounds DataFrame to a dictionary format suitable for YAML.

    This function processes a pandas DataFrame containing various quality metrics for each feature
    (e.g., `sigma_lower_bound`, `sigma_upper_bound`, `avg`, `std`, `iqr_lower_bound`, etc.) and
    converts it into a dictionary structure that can be serialized into YAML format.

    Parameters
    ----------
    quality_bounds_df : pandas.DataFrame
        A DataFrame containing quality metrics, with columns for each metric
        (e.g., 'sigma_lower_bound', 'sigma_upper_bound', 'avg', 'std', 'iqr_lower_bound', 'iqr_upper_bound', etc.).
        The index of the DataFrame represents the features (e.g., channels or other metrics).

    Returns
    -------
    dict
        A dictionary where each key is a column from the DataFrame, and each value is another dictionary
        containing the following keys:
        - 'sigma_range': A list containing the `sigma_lower_bound` and `sigma_upper_bound` values.
        - 'mean': The `avg` value.
        - 'std': The `std` value.
        - 'iqr_range': A list containing the `iqr_lower_bound` and `iqr_upper_bound` values.
        - 'q1': The first quartile value (`q1`).
        - 'q3': The third quartile value (`q3`).
        - 'median': The median value.
    """
    # Convert the DataFrame to a dictionary suitable for YAML
    bounds_dict = {}
    quality_bounds_dict = quality_bounds_df.to_dict()
    for col in quality_bounds_df.columns:
        bounds_dict[col] = {
            'sigma_range': [quality_bounds_dict[col]['sigma_lower_bound'],
                            quality_bounds_dict[col]['sigma_upper_bound']],
            'mean': quality_bounds_dict[col]['avg'],
            'std': quality_bounds_dict[col]['std'],
            'iqr_range': [quality_bounds_dict[col]['iqr_lower_bound'], quality_bounds_dict[col]['iqr_upper_bound']],
            'q1': quality_bounds_dict[col]['q1'],
            'q3': quality_bounds_dict[col]['q3'],
            'median': quality_bounds_dict[col]['median']
        }
    return bounds_dict

def save_quality_reference_to_yaml(quality_bounds_df, output_dir, filename="quality_reference.yaml", overwrite=False):
    """
    Save the quality reference bounds to a YAML file.

    Parameters
    ----------
    quality_bounds_df : pd.DataFrame
        DataFrame containing the quality bounds for each metric.
    output_dir : str or Path
        Directory where the quality reference YAML file will be saved.
    filename : str, optional
        The name of the YAML file. Default is "quality_reference.yaml".
    overwrite : bool, optional
        Whether to overwrite the existing file. Default is False, which will not overwrite existing files.
    Returns
    -------
    str
        The full path to the saved YAML file.
    """
    # Convert the DataFrame to a dictionary suitable for YAML
    bounds_dict = convert_to_yaml(quality_bounds_df)

    # Make sure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the dictionary to a YAML file
    yaml_path = output_dir / filename

    if yaml_path.exists() and not overwrite:
        raise FileExistsError(f"The file {yaml_path} already exists. Set 'overwrite=True' to overwrite.")

    with open(yaml_path, 'w') as file:
        yaml.dump(bounds_dict, file, default_flow_style=False)

    clogger.info(f"Quality reference YAML saved to: {yaml_path}")
    return str(yaml_path)


def is_bids_dataset(dataset_path: Union[Path, str], file_suffix: str = '.fif') -> bool:
    """
    Check if the given dataset path is a BIDS dataset.

    A BIDS dataset typically includes:
    - A 'sub-*' subject folder.
    - Optionally, a 'ses-*' session folder.
    - A 'meg' folder containing the actual MEG data files.
    - Required BIDS metadata files:
      - 'participants.tsv' file.
      - 'dataset_description.json' file.

    Parameters
    ----------
    dataset_path : Path or str
        The path to the dataset directory to check.

    file_suffix : str, optional
        The file extension to look for in the data files (default is '.fif').

    Returns
    -------
    bool
        True if the dataset follows BIDS structure, False otherwise.
    """
    dataset_path = Path(dataset_path)

    # Check for required BIDS metadata files in the root of the dataset
    dataset_description = dataset_path / 'dataset_description.json'
    participants_file = dataset_path / 'participants.tsv'

    if not dataset_description.exists() or not participants_file.exists():
        return False  # Missing essential BIDS metadata files

    # Check if the dataset contains a 'sub-*' directory (indicating BIDS structure)
    subject_dirs = list(dataset_path.glob('sub-*'))

    if not subject_dirs:
        return False  # No subject directories, not a BIDS dataset

    # For each subject directory, check if it contains the 'meg' folder
    for subject_dir in subject_dirs:
        # First, check directly in the subject directory for 'meg'
        meg_dir = subject_dir / 'meg'
        if meg_dir.exists() and any(meg_dir.glob(f'*{file_suffix}')):  # Check for the presence of .fif files
            return True

        # If no 'meg' folder directly under subject, check for 'ses-*' session directories
        session_dirs = list(subject_dir.glob('ses-*'))
        for session_dir in session_dirs:
            meg_dir_in_session = session_dir / 'meg'
            if meg_dir_in_session.exists() and any(meg_dir_in_session.glob(f'*{file_suffix}')):
                return True

    return False


def process_meg_quality(dataset_paths, data_type, file_suffix='.fif', n_jobs=-1,
                        output_dir="quality_ref", update_reference=False, device_name="new_device", overwrite=False):
    """
    Process the quality metrics for a list of MEG files, compute the quality reference bounds,
    and save them to a YAML file.

    Parameters
    ----------
    dataset_paths : list of str
        List of paths to the MEG datasets (either BIDS or Raw format).
    file_suffix : str
        The suffix of the file to read (e.g., '.fif'). Default is '.fif'.
    data_type : DATA_TYPE
        The data type for the quality metrics, e.g., 'opm' or 'squid'.
    n_jobs : int, optional
        Number of parallel jobs to use for metric computation. Default is -1 (use all available CPUs).
    output_dir : str, optional
        Directory where the quality reference YAML file will be saved. Default is "quality_ref".
    update_reference : bool, optional
        If True, update the reference YAML file in the msqms library. Default is False.
    device_name : str, optional
        The device name for the reference YAML file. Default is "new_device".
    overwrite : bool, optional
        Whether to overwrite the existing file. Default is False, which will not overwrite existing files.
    Returns
    -------
    str
        Path to the generated quality reference YAML file.
    """
    all_metrics_list = []

    # Iterate over the list of dataset paths
    for dataset_path in dataset_paths:
        dataset_metrics_list = []
        # Read the dataset files and extract raw MEG files
        # Check if the dataset is a BIDS dataset or a Raw dataset
        if is_bids_dataset(dataset_path, file_suffix):
            # Process BIDS datasets
            raw_list = read_raw_bids_dataset(Path(dataset_path))
        else:
            # Process Raw datasets
            raw_list = read_raw_dataset(dataset_path, file_suffix=file_suffix)

        for meg_filename in raw_list:
            # Calculate metrics for each MEG file
            metrics_df = calculate_quality_metrics(meg_filename, data_type=data_type, n_jobs=n_jobs)
            if metrics_df is not None:
                # For each subject, append the metrics DataFrame
                dataset_metrics_list.append(metrics_df)

        # Combine all subjects' metrics(from one dataset) into one DataFrame
        dataset_metrics_df = pd.concat(dataset_metrics_list, axis=0)

        # Calculate aggregated statistics for the combined dataset
        dataset_metrics_df.loc['avg'] = dataset_metrics_df.mean(axis=0)
        dataset_metrics_df.loc['std'] = dataset_metrics_df.std(axis=0)
        dataset_metrics_df.loc['q1'] = dataset_metrics_df.quantile(0.25)
        dataset_metrics_df.loc['median'] = dataset_metrics_df.median()
        dataset_metrics_df.loc['q3'] = dataset_metrics_df.quantile(0.75)

        all_metrics_list.append(dataset_metrics_df)

    # Combine all datasets' metrics into one DataFrame.
    # method1:
    # Initialize with the first dataset's metrics
    avg_dataset_df = all_metrics_list[0]
    # Accumulate the results of all datasets
    for dataset_df in all_metrics_list[1:]:
        avg_dataset_df = avg_dataset_df + dataset_df
    # Average across all datasets
    avg_all_metrics_df = avg_dataset_df / len(all_metrics_list)

    # method2:
    all_metrics_df = pd.concat(all_metrics_list, axis=0)
    avg_all_metrics_df2 = all_metrics_df.groupby(all_metrics_df.index).mean()

    print("Check Consistency", avg_all_metrics_df.equals(avg_all_metrics_df2))

    # Calculate the quality reference bounds based on the computed metrics
    quality_bounds_df = obtain_quality_ranges(avg_all_metrics_df)

    # Optionally, update the quality reference in the OPQMC library
    if update_reference:
        update_quality_reference_file(quality_bounds_df, device_name=device_name, overwrite=overwrite)

    # Save the quality reference bounds to a YAML file
    yaml_path = save_quality_reference_to_yaml(quality_bounds_df, output_dir, overwrite=True)

    return yaml_path


def update_quality_reference_file(quality_reference_df, device_name, overwrite=False):
    """
    Updates the existing quality reference file for a given device name in the msqms library.

    Automatically determines the msqms installation directory.

    Parameters
    ----------
    quality_reference_df : pd.DataFrame
        The DataFrame containing the calculated quality reference bounds.
    device_name : str
        The name of the device, which will be used to name the YAML file (e.g., 'opm', 'squid').
    overwrite : bool, optional
        Whether to overwrite the existing file. Default is False, which will not overwrite existing files.

    Returns
    -------
    str
        The path to the updated quality reference YAML file.
    """
    try:
        # Get the directory of quality reference yaml files.
        quality_reference_dir = Path(__file__).parent

        print("qr_dir", quality_reference_dir)
        # Define the path for the YAML file inside msqms library directory
        quality_reference_file = quality_reference_dir / f"{device_name}_quality_reference.yaml"

        if quality_reference_file.exists() and not overwrite:
            raise FileExistsError(f"The file {quality_reference_file} already exists. Set 'overwrite=True' to overwrite.")

        # Save the YAML data to the file
        quality_reference_dict = convert_to_yaml(quality_reference_df)
        with open(quality_reference_file, 'w') as file:
            yaml.dump(quality_reference_dict, file, default_flow_style=False)

        clogger.info(f"Quality reference for {device_name} has been updated and saved to: {quality_reference_file}")
        return quality_reference_file
    except FileExistsError as e:
        clogger.error(f"error updating quality reference for {device_name}:{e}")



def list_existing_quality_references():
    """
    List all existing quality reference YAML files in the installed msqms library.

    This function searches the 'quality_reference' directory within the msqms installation
    for YAML files following the '<device_name>_quality_reference.yaml' naming pattern.

    Returns
    -------
    list of tuple
        A list where each tuple contains the device name and the corresponding YAML file path.

    Raises
    ------
    ImportError
        If the msqms library is not installed or cannot be found.
    """

    # Define the path to the 'quality_reference' directory within msqms
    quality_reference_dir = Path(__file__).parent

    # Check if the 'quality_reference' directory exists
    if not quality_reference_dir.exists():
        clogger.info(f"No quality reference directory found at {quality_reference_dir}.")
        return []

    # Find all YAML files matching the pattern '<device_name>_quality_reference.yaml'
    yaml_files = list(quality_reference_dir.glob("*_quality_reference.yaml"))

    if not yaml_files:
        clogger.info(f"No quality reference YAML files found in {quality_reference_dir}.")
        return []

    # Extract device names and file paths
    quality_references = []
    for yaml_file in yaml_files:
        # Extract the device name from the file name
        device_name = yaml_file.stem.replace("_quality_reference", "")
        quality_references.append((device_name, str(yaml_file)))

    # clogger.info the found quality reference files
    clogger.info(f"Found the following quality reference files in {quality_reference_dir}:")
    for device, path in quality_references:
        clogger.info(f"Device: {device}, File: {path}")

    return quality_references


if __name__ == '__main__':
    bids_dir = Path(r'C:\Data\Datasets\OPM-COG.v1')  # BIDS格式的数据集，计算质控区间；
    raw_dir = Path(r"C:\Data\Datasets\OPM-Artifacts")  # Raw格式的数据集，计算质控区间；

    # update_quality_ref(dataset_path=bids_dir, output_dir='.')

    # 指定数据集路径，bids or raw，读取得到fif文件

    # 质控区间计算

    # 质控区间更新方式：根据数据集，生成一个全新的质控参考区间；需要并行计算；

    # 质控区间生成yaml文件，输出保存yaml文件的路径；

    # 函数：将新得到的质控区间，更新到msqms库内部安装目录下yaml文件中，指定质控参考yaml名称；
    test_pd = pd.DataFrame({"A": [1, 2, 3]})
    update_quality_reference_file(test_pd, device_name="opm2", overwrite=True)

    # 工具函数：获取目前已有的质控区间yaml，方便用户指定不同的设备名称来获取。比如说opm、squid、quspin,quanmag,ctf等；
    # list_existing_quality_references()
