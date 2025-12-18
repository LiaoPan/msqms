# MSQMs: Magnetometers Quality Control Tool
MEG Signal Quality Metrics (MSQMs) is a fully automated quality control tool for OPM-MEG and SQUID-MEG.

[![PyPI - Version](https://img.shields.io/pypi/v/msqms?color)](https://badge.fury.io/msqms)
[![Documentation](https://readthedocs.org/projects/msqms/badge/?version=latest)](https://keras-complex.readthedocs.io/)
[![PyPI Versions](https://img.shields.io/pypi/pyversions/msqms.svg?color)](https://pypi.python.org/pypi/msqms?color=deepgreen)
![PyPI - Downloads](https://img.shields.io/pypi/dm/msqms)


[//]: # ([![Build Status]&#40;https://travis-ci.org/JesperDramsch/msqms.svg?branch=master?color&#41;]&#40;https://travis-ci.org/JesperDramsch/msqms&#41;)
[//]: # ([![PyPI Status]&#40;https://img.shields.io/pypi/status/msqms.svg?color&#41;]&#40;https://pypi.python.org/pypi/msqms&#41;)

## Installation
```bash
pip install msqms
```

## Using CLI Tool to Generate Summary Quality Control Reports

### Command Description

The new `msqms_summary` command is used to generate summary quality control reports for multiple MEG files. This command automatically traverses all matching files in the specified directory.

### Basic Usage

```bash
msqms_summary -i ./data -o ./reports -t opm
```

### Parameter Explanation

- `-i, --input`: Input directory path (required, contains the directory with MEG files)
- `-s, --suffix`: File suffix (optional, default is '.fif', e.g., '.fif', '.ds')
- `-o, --outdir`: Output directory (required, default is the current directory)
- `-t, --data_type`: Data type (required, choices are 'opm' or 'squid')
- `-n, --report_name`: Summary report filename (optional, default is 'summary_report')
- `-r, --recursive`: Recursively search subdirectories (optional flag)

### Usage Examples

#### Example 1: Process All .fif Files in the Directory (default)

```bash
msqms_summary -i ./data -o ./quality_reports -t opm
```

#### Example 2: Process Files with Specified Suffix

```bash
# Process .ds files
msqms_summary -i ./data -s .ds -o ./output -t squid

# Process .fif files (explicitly specified)
msqms_summary -i ./data -s .fif -o ./output -t opm
```

#### Example 3: Recursively Search Subdirectories

```bash
# Recursively search for all .fif files in subdirectories
msqms_summary -i ./data -r -o ./reports -t opm
```

#### Example 4: Custom Report Name

```bash
msqms_summary -i ./data -o ./output -t opm -n my_summary_report
```

#### Example 5: Complete Example

```bash
# Recursively search for all .fif files and generate a custom-named summary report
msqms_summary -i /path/to/meg/data -r -s .fif -o /path/to/output -t opm -n batch_quality_report
```

### Output Files

After executing the command, the following will be generated:

1. **Summary Report**: `{report_name}.html` - Contains statistical information and visual charts for all files
2. **Individual Reports**: Each file will generate a corresponding `{filename}.report.html`
3. **Visual Images**: Each file will create a corresponding `{filename}.imgs/` directory

### Summary Report Features

The generated summary report includes:

- **Summary Tab**:
  - Summary statistics (total number of files, average scores, standard deviations, etc.)
  - MSQM score distribution charts
  - Distribution charts for various metrics (Time Metrics, Frequency Metrics, Entropy Metrics, Fractal Metrics, Artifacts)

- **Individual Reports Tab**:
  - A list of all files displaying filenames and scores
  - Clicking on any file allows viewing the detailed report below
  - Quality level badges (Excellent/Good/Fair/Poor/Bad)

### Help Information

To view the complete help information:

```bash
msqms_summary --help
```

---

## Using CLI Tool to Generate Single Quality Control Report

   Using `msqms_report` to Generate Quality Control (QC) Reports for MEG Data

### Command Description

The `msqms_report` command is used to generate a Quality Control (QC) report for a specified MEG data file. This report includes various quality metrics calculated from the MEG data, such as signal quality, noise levels, and other key statistics. The report is saved in the specified output directory. Users must specify the type of MEG data, which can be either `opm` or `squid`.

### Basic Usage

```bash
msqms_report -f ./auditory_raw.fif -o ./ -t opm
```

### Parameter Explanation

- `--file, -f`:  
  The path to the MEG file required for quality assessment. This option is mandatory.

- `--outdir, -o`:  
  The directory where the generated quality report will be saved. This option is mandatory. Default is the current directory (`.`).

- `--data_type, -t`:  
  The type of MEG data. Choose either `opm` or `squid`. This option is mandatory.

### Usage Example

To generate a QC report for a `data.fif` file and save the report in the `reports/` directory, use the following command:

```bash
msqms_report --file data/data.fif --outdir reports/ --data_type opm
```

This command processes the `data.fif` MEG file, calculates the relevant quality metrics for `opm` data, and generates an HTML quality control report in the `reports/` directory. The generated report will be named `data.report.html`.

### Help Information

To see the help documents for the command:

```bash
msqms_report -h
```

## Bug reports
Please use the [GitHub issue tracker](https://github.com/LiaoPan/msqms/issues) to report bugs.


## Interested in Contributing?
Please read our contributing guide.

## Citation
When using MSQMs, please include the following citation:


