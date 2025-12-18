Command-Line Tools
========================

This documentation provides the usage for the msqms command-line tools, which, after the installation of MSQMs, can be used to evaluate the quality of single or multiple MEG (magnetoencephalography) data files. These tools allow for the calculation, updating, and listing of quality reference bounds, as well as the generation of comprehensive quality control reports for MEG datasets. The tools support efficient data quality management by providing automated workflows for reference calculation, updating existing quality reference files, and generating detailed reports based on quality metrics.
:doc:`See in details <../apis/msqms.cli.workflow>`

1. **msqms_report**
-----------------------------------------------------------------------

Generate a Quality Control (QC) Report for MEG Data

**Usage:**

.. code-block::

    msqms_report --file <file_path> --outdir <output_directory> --data_type <data_type> [OPTIONS]

.. code-block:: bash

     # for windows
     $ msqms_report.exe -f .\auditory_raw.fif -o .\ -t opm

     # for linux
     $ msqms_report -f ./auditory_raw.fif -o ./ -t opm

See help documents:

.. code-block:: bash

     # for windows
     $ msqms_report.exe -h

     # for linux
     $ msqms_report -h


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

    msqms_report --file data/data.fif --outdir reports/ --data_type opm

The above command will process the `data.fif` MEG file, calculate the relevant quality metrics for `opm` data, and generate an HTML quality control report in the `reports/` directory. The generated report will be named `data.report.html`.



2. **msqms_quality_ref_cal**
--------------------------------------------------------------------------

Compute and update quality reference bounds

**Usage:**

.. code-block::

    msqms_quality_ref_cal <dataset_paths> [OPTIONS]

**Description:**

This command is used to compute and update the quality reference bounds based on multiple MEG datasets.
It will calculate quality metrics for each dataset and generate a quality reference YAML file that contains the reference bounds (avg, std, q1, median, q3).
If needed, it can also update an existing quality reference file in the `msqms` library.


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
    If set, this flag will update the reference quality YAML file in the MSQMs library. The specified dataset's quality metrics will be used to update the existing quality reference file for the corresponding device.

- `--device-name, -d <device_name>`
    The device name associated with the quality reference file (e.g., `opm`, `squid`, etc.). This will be used to determine the filename of the YAML file (`<device_name>_quality_reference.yaml`). The default is `opm`.

- `--overwrite, -w`
    If set, this flag will overwrite the existing quality reference file in the MSQMs library. By default, the tool will not overwrite existing files.

Description
-----------
This tool allows users to process one or more MEG datasets, calculate the quality metrics for each, and generate a quality reference YAML file that summarizes the quality bounds (average, standard deviation, quantiles, etc.) across the datasets. If the `--update-reference` option is used, the quality reference YAML file for a specific device in the MSQMs library will be updated using the newly computed metrics. The `--overwrite` option allows the tool to replace the existing reference file.

When the `--update-reference` option is specified, the tool updates the reference YAML file for the corresponding device (e.g., `opm` or `squid`) in the MSQMs library. By default, the `--device-name` option is set to `opm`, but you can change it to any valid device name.

Examples
--------

1. Compute the quality reference for a single dataset and save it to the default output directory:

.. code-block::

    msqms_quality_ref_cal --dataset_paths data/ --data_type opm --output_dir quality_ref --overwrite


3. **msqms_quality_ref_update**
------------------------------------------------------------------

Update existing quality reference

**Usage:**

.. code-block::

    msqms_quality_ref_update [OPTIONS]

**Description:**

This command is used to update an existing quality reference file in the `msqms` library with a new YAML file.
It will replace the current reference file with the new one, provided that the file path is specified.

**Options:**

- `--device_name, -d <device_name>`
    The device name associated with the quality reference file (e.g., `opm`, `squid`, etc.). This will be used to determine the filename of the YAML file (`<device_name>_quality_reference.yaml`).

- `--quality_reference_file, -q <file_path>`
    The path to the quality reference YAML file that contains the new quality metrics. This file will be used to update the `<device_name>_quality_reference.yaml` file located in the MSQMs library.

- `--overwrite, -w`
    If provided, the command will overwrite the existing quality reference file. Without this option, the tool will not update the file if it already exists.

**Example:**

To update the quality reference for the `opm` device:

.. code-block::

    msqms_quality_ref_update -q quality_ref/opm_quality_reference.yaml -d opm


4. **msqms_quality_ref_list**
----------------------------------------------------------------

List existing quality references

**Usage:**

.. code-block::

    msqms_quality_ref_list [OPTIONS]

**Description:**

This command is used to list all existing quality reference files in the `msqms` library. It will display the device name and the file path for each reference file.

**Example:**

To list all existing quality reference files:

.. code-block::

    msqms_quality_ref_list

5. **msqms_summary**
----------------------------------------------------------------
Using CLI Tool to Generate Summary Quality Control Reports

**Description**

The new ``msqms_summary`` command is used to generate summary quality control reports for multiple MEG files. This command automatically traverses all matching files in the specified directory.

**Usage**

.. code-block:: bash

    msqms_summary -i ./data -o ./reports -t opm

**Parameter Explanation**

- ``-i, --input``: Input directory path (required, contains the directory with MEG files)
- ``-s, --suffix``: File suffix (optional, default is '.fif', e.g., '.fif', '.ds')
- ``-o, --outdir``: Output directory (required, default is the current directory)
- ``-t, --data_type``: Data type (required, choices are 'opm' or 'squid')
- ``-n, --report_name``: Summary report filename (optional, default is 'summary_report')
- ``-r, --recursive``: Recursively search subdirectories (optional flag)

**Examples**

Example 1: Process All .fif Files in the Directory (default)

.. code-block:: bash

    msqms_summary -i ./data -o ./quality_reports -t opm

Example 2: Process Files with Specified Suffix

.. code-block:: bash

    # Process .ds files
    msqms_summary -i ./data -s .ds -o ./output -t squid

    # Process .fif files (explicitly specified)
    msqms_summary -i ./data -s .fif -o ./output -t opm

Example 3: Recursively Search Subdirectories

.. code-block:: bash

    # Recursively search for all .fif files in subdirectories
    msqms_summary -i ./data -r -o ./reports -t opm

Example 4: Custom Report Name

.. code-block:: bash

    msqms_summary -i ./data -o ./output -t opm -n my_summary_report

Example 5: Complete Example

.. code-block:: bash

    # Recursively search for all .fif files and generate a custom-named summary report
    msqms_summary -i /path/to/meg/data -r -s .fif -o /path/to/output -t opm -n batch_quality_report

**Output Files**


After executing the command, the following will be generated:

1. **Summary Report**: ``{report_name}.html`` - Contains statistical information and visual charts for all files
2. **Individual Reports**: Each file will generate a corresponding ``{filename}.report.html``
3. **Visual Images**: Each file will create a corresponding ``{filename}.imgs/`` directory

**Summary Report Features**

The generated summary report includes:

- **Summary Tab**:
  - Summary statistics (total number of files, average scores, standard deviations, etc.)
  - MSQM score distribution charts
  - Distribution charts for various metrics (Time Metrics, Frequency Metrics, Entropy Metrics, Fractal Metrics, Artifacts)

- **Individual Reports Tab**:
  - A list of all files displaying filenames and scores
  - Clicking on any file allows viewing the detailed report below
  - Quality level badges (Excellent/Good/Fair/Poor/Bad)

