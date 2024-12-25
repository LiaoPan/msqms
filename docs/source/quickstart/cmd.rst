Command-Line Tools
========================

This documentation provides the usage for the opmqc command-line tools, which, after the installation of OPMQC, can be used to evaluate the quality of single or multiple MEG (magnetoencephalography) data files. These tools allow for the calculation, updating, and listing of quality reference bounds, as well as the generation of comprehensive quality control reports for MEG datasets. The tools support efficient data quality management by providing automated workflows for reference calculation, updating existing quality reference files, and generating detailed reports based on quality metrics.
:doc:`See in details <../apis/opmqc.cli.workflow>`

1. **opmqc_report**
-----------------------------------------------------------------------

Generate a Quality Control (QC) Report for MEG Data

**Usage:**

.. code-block::

    opmqc_report --file <file_path> --outdir <output_directory> --data_type <data_type> [OPTIONS]

.. code-block:: bash

     # for windows
     $ opmqc_report.exe -f .\auditory_raw.fif -o .\ -t opm

     # for linux
     $ opmqc_report -f ./auditory_raw.fif -o ./ -t opm

See help documents:

.. code-block:: bash

     # for windows
     $ opmqc_report.exe -h

     # for linux
     $ opmqc_report -h


**Description:**

This command generates a Quality Control (QC) report for a given MEG data file. The report includes various quality metrics calculated from the MEG data, such as signal quality, noise levels, and other key statistics.
The report is saved in the specified output directory. The user must specify the type of MEG data, which can be either `opm` or `squid`.

**Options:**

- `--file, -f`
    The path to the MEG file required for quality assessment. This option is **required**.

- `--outdir, -o`
    The directory where the generated quality report will be saved. This option is **required**. Default is the current directory (`.`).

- `--data_type, -t`
    The type of MEG data. Choose either `opm` or `squid`. This option is **required**.

**Example:**

To generate a QC report for a `data.fif` file and save the report in the `reports/` directory:

.. code-block::

    opmqc_report --file data/data.fif --outdir reports/ --data_type opm

The above command will process the `data.fif` MEG file, calculate the relevant quality metrics for `opm` data, and generate an HTML quality control report in the `reports/` directory. The generated report will be named `data.report.html`.



2. **opmqc_quality_ref_cal**
--------------------------------------------------------------------------

Compute and update quality reference bounds

**Usage:**

.. code-block::

    opmqc_quality_ref_cal <dataset_paths> [OPTIONS]

**Description:**

This command is used to compute and update the quality reference bounds based on multiple MEG datasets.
It will calculate quality metrics for each dataset and generate a quality reference YAML file that contains the reference bounds (avg, std, q1, median, q3).
If needed, it can also update an existing quality reference file in the `opmqc` library.


**Options:**

- `--dataset_paths, -p <dataset_path_1> <dataset_path_2> ...`
    The paths to one or more datasets. This can be a list of directories or files containing MEG data. The datasets will be processed to calculate quality metrics.

    dataset_format: BIDS Format or Raw Format

    The Raw format of the dataset. Can be one of:
    - 'format1': Dataset with subject subdirectories.
    - 'format2': Dataset with raw data files directly in the main directory.

    - 'format1' datasets dir/
        - sub01/
            - *.fif (or *.ds)
        - sub02/
            - *.fif (or *.ds)
        - ...

    - 'format2' datasets dir/
        - *.fif (or *.ds)

- `--file-suffix, -s <suffix>`
    The file suffix for the MEG files to process (default is `.fif`). If your MEG files have a different extension, specify it here.

- `--data-type, -t <data_type>`
    The data type for the quality metrics. Choose between `opm` or `squid` (default is `opm`). This option specifies the type of MEG data.

- `--n-jobs, -n <num>`
    The number of parallel jobs to use for metric computation (default is `-1`, which uses all available CPUs).

- `--output-dir, -o <dir>`
    The directory where the resulting YAML file containing the computed quality reference bounds will be saved (default is `quality_ref`).

- `--update-reference, -u`
    If set, this flag will update the reference quality YAML file in the OPMQC library. The specified dataset's quality metrics will be used to update the existing quality reference file for the corresponding device.

- `--device-name, -d <device_name>`
    The device name associated with the quality reference file (e.g., `opm`, `squid`, etc.). This will be used to determine the filename of the YAML file (`<device_name>_quality_reference.yaml`). The default is `opm`.

- `--overwrite, -w`
    If set, this flag will overwrite the existing quality reference file in the OPMQC library. By default, the tool will not overwrite existing files.

Description
-----------
This tool allows users to process one or more MEG datasets, calculate the quality metrics for each, and generate a quality reference YAML file that summarizes the quality bounds (average, standard deviation, quantiles, etc.) across the datasets. If the `--update-reference` option is used, the quality reference YAML file for a specific device in the OPMQC library will be updated using the newly computed metrics. The `--overwrite` option allows the tool to replace the existing reference file.

When the `--update-reference` option is specified, the tool updates the reference YAML file for the corresponding device (e.g., `opm` or `squid`) in the OPMQC library. By default, the `--device-name` option is set to `opm`, but you can change it to any valid device name.

Examples
--------

1. Compute the quality reference for a single dataset and save it to the default output directory:

.. code-block::

    opmqc_quality_ref_cal --dataset_paths data/ --data_type opm --output_dir quality_ref --overwrite


3. **opmqc_quality_ref_update**
------------------------------------------------------------------

Update existing quality reference

**Usage:**

.. code-block::

    opmqc_quality_ref_update [OPTIONS]

**Description:**

This command is used to update an existing quality reference file in the `opmqc` library with a new YAML file.
It will replace the current reference file with the new one, provided that the file path is specified.

**Options:**

- `--device_name, -d <device_name>`
    The device name associated with the quality reference file (e.g., `opm`, `squid`, etc.). This will be used to determine the filename of the YAML file (`<device_name>_quality_reference.yaml`).

- `--quality_reference_file, -q <file_path>`
    The path to the quality reference YAML file that contains the new quality metrics. This file will be used to update the `<device_name>_quality_reference.yaml` file located in the OPMQC library.

- `--overwrite, -w`
    If provided, the command will overwrite the existing quality reference file. Without this option, the tool will not update the file if it already exists.

**Example:**

To update the quality reference for the `opm` device:

.. code-block::

    opmqc_quality_ref_update -q quality_ref/opm_quality_reference.yaml -d opm


4. **opmqc_quality_ref_list**
----------------------------------------------------------------

List existing quality references

**Usage:**

.. code-block::

    opmqc_quality_ref_list [OPTIONS]

**Description:**

This command is used to list all existing quality reference files in the `opmqc` library. It will display the device name and the file path for each reference file.

**Example:**

To list all existing quality reference files:

.. code-block::

    opmqc_quality_ref_list



