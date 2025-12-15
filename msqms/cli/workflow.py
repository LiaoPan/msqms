# -*- coding: utf-8 -*-
import click
import yaml
import pandas as pd
from pathlib import Path
from msqms.reports.report import gen_quality_report
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
              help="If set, will update the reference quality YAML file in the OPQMC library")
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
