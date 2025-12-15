# -*- coding: utf-8 -*-
"""
Read Raw datasets
DatasetFormat:
Format 1 (Subject directories):
- datasets dir/
    - sub01/ *.fif
    - sub02/ *.fif
    ...
    - sub03/ *.fif

or Format 2 (Single directory):
- datasets dir/
    - *.fif

Get all fif file,and return the raw object lists.
"""

import os


def read_raw_dataset(dataset_dir, file_suffix='.fif', dataset_format=None):
    """
    Finds raw data file paths with the specified suffix from subdirectories or directly within a specified dataset directory.
    It supports both standard MNE formats (.fif) and CTF formats (.ds).

    This function checks that the dataset follows one of two possible structures:
    - datasets dir/
        - sub01/
            - *.fif (or *.ds)
        - sub02/
            - *.fif (or *.ds)
        - ...
    OR
    - datasets dir/
        - *.fif (or *.ds)

    Parameters
    ----------
    dataset_dir : str
        The path to the main directory containing raw data files. This can either be a directory containing
        subdirectories for each subject or just a collection of raw data files in the main directory.

    file_suffix : str, optional
        The suffix of the files to find (default is '.fif'). Can be '.fif' for MNE datasets or '.ds' for CTF datasets.

    dataset_format : str, optional
        The format of the dataset. Can be one of:
        - 'format1': Dataset with subject subdirectories.
        - 'format2': Dataset with raw data files directly in the main directory.
        If not provided, the function will attempt to detect the format automatically.

    Returns
    -------
    raw_list : list of str
        A list of file paths of the raw data files found in the subdirectories or directly in the dataset directory.

    Raises
    ------
    ValueError
        If the dataset directory doesn't follow the expected structure or contains unsupported file types.

    Notes
    -----
    This function assumes that the dataset directory contains either:
    - subdirectories for each subject, each containing one or more raw data files with the specified suffix,
    OR
    - just raw data files directly within the main directory.

    Example
    -------
    dataset_dir = "/path/to/dataset"
    raw_list = read_raw_dataset(dataset_dir, file_suffix='.fif', dataset_format='format1')

    # Print the file paths
    for path in raw_list:
        print(path)
    """
    raw_list = []

    # Check if the dataset directory exists
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"The specified dataset directory {dataset_dir} is not valid.")

    # Automatically detect the dataset format if not provided
    if dataset_format is None:
        subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        if subdirs:
            dataset_format = 'format1'  # Detected Format 1: Subject subdirectories. BIDS Format
        else:
            dataset_format = 'format2'  # Detected Format 2: Files directly in the main directory. Raw Format

    # Handle 'format1' (dataset_dir contains subdirectories)
    if dataset_format == 'format1':
        subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        for subdir in subdirs:
            subdir_path = os.path.join(dataset_dir, subdir)
            print(f"Checking subdirectory: {subdir_path}")  # Debugging line
            raw_file_found = False
            for file in os.listdir(subdir_path):
                if file.endswith(file_suffix):
                    raw_file_found = True
                    file_path = os.path.join(subdir_path, file)
                    raw_list.append(file_path)

            # If no file with the expected suffix is found in the subdir, raise an error
            if not raw_file_found:
                print(f"No file with suffix {file_suffix} found in {subdir_path}")  # Debugging line
                raise ValueError(f"No file with suffix {file_suffix} found in subject directory: {subdir}")

    # Handle 'format2' (dataset_dir contains raw data files directly)
    elif dataset_format == 'format2':
        for file in os.listdir(dataset_dir):
            if file.endswith(file_suffix):
                file_path = os.path.join(dataset_dir, file)
                raw_list.append(file_path)

    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}; Supported formats: 'format1', 'format2'")

    # If no raw data files were found, raise an error
    if not raw_list:
        raise ValueError(f"No raw data files found in {dataset_dir}")

    return raw_list


if __name__ == "__main__":
    dataset_dir = r"C:\Data\Datasets\OPM-Artifacts"
    raw_list = read_raw_dataset(dataset_dir, file_suffix='.fif', dataset_format='format2')
    # Print the raw data file paths
    for path in raw_list:
        print(path)