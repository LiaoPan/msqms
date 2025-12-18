# -*- coding: utf-8 -*-
import click
import yaml
import pandas as pd
import os
from pathlib import Path
from msqms.reports.report import gen_quality_report,gen_summary_quality_report
from msqms.quality_reference import list_existing_quality_references
from msqms.quality_reference import update_quality_reference_file, process_meg_quality

# config click
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    '--file', '-f', type=click.Path(exists=True, dir_okay=True, readable=True), required=True,
    help='The MEG file required for quality assessment.')
@click.option(
    '--outdir', '-o', type=click.Path(file_okay=False, writable=True), required=True, default='.', show_default=True,
    help='The output directory for the quality report.')
@click.option(
    '--data_type', '-t', type=click.Choice(['opm', 'squid'], case_sensitive=False), required=True,
    help="The type of MEG data. Choose from ['opm', 'squid'].")
def generate_qc_report(file, outdir, data_type):
    """
    Generate a Quality Control (QC) Report for MEG data.

    This function processes a MEG data file and generates a quality control
    report in the specified output directory. The user need to specify the type
    of MEG data, either 'opm' or 'squid'.
    """
    filename = Path(file).stem + '.report'
    gen_quality_report([file], outdir=outdir, report_fname=filename, data_type=data_type, ftype='html')
    


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--dataset_paths', '-p', multiple=True, type=click.Path(exists=True),
              help='The paths of datasets. Accepts multiple paths separated by spaces.')
@click.option('--file-suffix', '-s', default='.fif', help='File suffix for the MEG files (default is .fif)')
@click.option('--data-type', '-t', default='opm', type=click.Choice(['opm', 'squid']),
              help="Data type for the quality metrics (default is 'opm')")
@click.option('--n-jobs', '-n', default=-1, type=int, help="Number of parallel jobs (default is -1, use all CPUs)")
@click.option('--output-dir', '-o', default='quality_ref', help="Directory where the YAML file will be saved")
@click.option('--update-reference', '-u', is_flag=True,
              help="If set, will update the reference quality YAML file in the MSQMs library")
@click.option('--device-name', '-d', default='opm',
              help="Device name for the YAML reference file (default is 'opm'). For example,<device_name>_quality_reference.yaml)")
@click.option('--overwrite', '-w', is_flag=True, help="If set, will overwrite the existing quality reference file")
def compute_and_update_quality_reference(dataset_paths, file_suffix, data_type, n_jobs,
                                         output_dir, update_reference, device_name, overwrite):
    """
    Command to process MEG quality metrics for a list of datasets and optionally update the reference YAML.
    In details, computing and updating the quality reference bounds based on multiple MEG datasets.
    """
    yaml_path = process_meg_quality(
        dataset_paths=dataset_paths,
        file_suffix=file_suffix,
        data_type=data_type,
        n_jobs=n_jobs,
        output_dir=output_dir,
        update_reference=update_reference,
        device_name=device_name,
        overwrite=overwrite
    )
    click.echo(f"Quality reference YAML saved to {yaml_path}")


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--quality_reference_file', '-q', type=click.Path(exists=True), help='Quality reference YAML file')
@click.option('--device_name', '-d', type=str, help="Device name for the YAML reference file (default is 'opm'). For "
                                                    "example,<device_name>_quality_reference.yaml)")
@click.option('--overwrite', '-w', is_flag=True, help="Overwrite the existing quality reference YAML file if it exists")
def update_quality_reference(device_name, quality_reference_file, overwrite):
    """
    Update the quality reference YAML file for a specific device in the msqms library.
    """
    try:
        # Load the quality reference data from the provided YAML file
        with open(quality_reference_file, 'r') as file:
            quality_reference_data = yaml.safe_load(file)

        # Convert the quality reference data to a DataFrame
        quality_reference_df = pd.DataFrame.from_dict(quality_reference_data, orient='index')

        # Update the quality reference file in the msqms library
        updated_yaml_path = update_quality_reference_file(quality_reference_df, device_name=device_name,
                                                          overwrite=overwrite)
        print(f"Quality reference for '{device_name}' has been updated and saved to: {updated_yaml_path}")
    except Exception as e:
        print(f"Error updating quality reference: {e}")


@click.command(context_settings=CONTEXT_SETTINGS)
def list_quality_references():
    """
    List all existing quality reference YAML files in the msqms library.
    """
    try:
        quality_references = list_existing_quality_references()
        if not quality_references:
            click.echo("No quality reference files found.")
            return

        click.echo("Existing Quality Reference Files:")
        for device, path in quality_references:
            click.echo(f"Device: {device}, File: {path}")
    except Exception as e:
        click.echo(f"Error listing quality reference files: {e}")


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    '--input', '-i', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True), required=True,
    help='The input directory containing MEG files to process.')
@click.option(
    '--suffix', '-s', type=str, default='.fif', show_default=True,
    help='File suffix/extension to search for (e.g., .fif, .ds). Default is .fif')
@click.option(
    '--outdir', '-o', type=click.Path(file_okay=False, writable=True), required=True, default='.', show_default=True,
    help='The output directory for the quality reports.')
@click.option(
    '--data_type', '-t', type=click.Choice(['opm', 'squid'], case_sensitive=False), required=True,
    help="The type of MEG data. Choose from ['opm', 'squid'].")
@click.option(
    '--report_name', '-n', type=str, default='summary_report', show_default=True,
    help='The name of the generated summary report file (without extension).')
@click.option(
    '--recursive', '-r', is_flag=True, default=False,
    help='Search for files recursively in subdirectories.')
def generate_summary_qc_report(input, suffix, outdir, data_type, report_name, recursive):
    """
    Generate a Summary Quality Control (QC) Report for multiple MEG data files.

    This function automatically finds all MEG files with the specified suffix in the input
    directory and processes them to generate:
    1. Individual quality control reports for each file
    2. A summary report with aggregated statistics and visualizations
    3. Distribution charts for MSQM scores and category metrics

    The summary report allows you to:
    - View aggregated statistics across all files
    - See distribution charts for quality scores
    - Easily switch between individual reports
    - Compare quality metrics across different files

    Examples:
        # Process all .fif files in a directory
        msqms_summary -i ./data -o ./reports -t opm

        # Process files with custom suffix
        msqms_summary -i ./data -s .ds -o ./reports -t squid

        # Recursively search in subdirectories
        msqms_summary -i ./data -r -o ./reports -t opm
    """
    # Ensure suffix starts with a dot
    if not suffix.startswith('.'):
        suffix = '.' + suffix
    
    input_path = Path(input)
    
    # Find all files with the specified suffix
    if recursive:
        # Recursive search
        meg_files = list(input_path.rglob(f'*{suffix}'))
    else:
        # Only search in the input directory (non-recursive)
        meg_files = [f for f in input_path.iterdir() 
                     if f.is_file() and f.suffix.lower() == suffix.lower()]
    
    # Sort files naturally
    meg_files = sorted(meg_files, key=lambda x: str(x).lower())
    
    if not meg_files:
        click.echo(f"Error: No files with suffix '{suffix}' found in {input_path}", err=True)
        click.echo(f"Search mode: {'recursive' if recursive else 'non-recursive'}", err=True)
        return
    
    click.echo(f"Found {len(meg_files)} file(s) with suffix '{suffix}' in {input_path}")
    if len(meg_files) <= 10:
        click.echo("Files to process:")
        for f in meg_files:
            click.echo(f"  - {f.name}")
    else:
        click.echo("Files to process (showing first 10):")
        for f in meg_files[:10]:
            click.echo(f"  - {f.name}")
        click.echo(f"  ... and {len(meg_files) - 10} more files")
    click.echo("")
    click.echo(f"Output directory: {outdir}")
    click.echo(f"Data type: {data_type}")
    click.echo(f"Summary report name: {report_name}")
    click.echo("")

    try:
        # Convert Path objects to strings
        meg_files_str = [str(f) for f in meg_files]
        
        summary_stats = gen_summary_quality_report(
            megfiles=meg_files_str,
            outdir=outdir,
            report_fname=report_name,
            data_type=data_type,
            ftype='html'
        )

        if summary_stats:
            click.echo("")
            click.echo("=" * 60)
            click.echo("Summary Report Generated Successfully!")
            click.echo("=" * 60)
            click.echo(f"Total files processed: {summary_stats['statistics']['total_files']}")
            click.echo(f"Average MSQM score: {summary_stats['statistics']['msqm_mean'] * 100:.2f}%")
            click.echo(f"MSQM score range: {summary_stats['statistics']['msqm_min'] * 100:.2f}% - {summary_stats['statistics']['msqm_max'] * 100:.2f}%")
            click.echo(f"MSQM score std: {summary_stats['statistics']['msqm_std'] * 100:.2f}%")
            click.echo("")
            click.echo(f"Summary report saved to: {Path(outdir) / f'{report_name}.html'}")
            click.echo(f"Individual reports saved to: {outdir}")
            click.echo("")
            click.echo("Open the summary report in a web browser to view:")
            click.echo("  - Aggregated statistics")
            click.echo("  - Distribution charts for quality scores")
            click.echo("  - Individual quality reports")
        else:
            click.echo("Error: Failed to generate summary report.", err=True)
    except Exception as e:
        click.echo(f"Error generating summary report: {e}", err=True)
        raise