# -*- coding: utf-8 -*-
import click
from pathlib import Path
from opmqc.reports.report import gen_quality_report

# config click
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--file', '-f', type=str, required=True, help='the fif file required for quality assessment.')
@click.option('--outdir', '-o', type=str, required=True, default='.', show_default=True, help='the output path of quality report.')
@click.option('--data_type', '-t', type=str, required=True, default='opm', show_default=True, help="the type of MEG data['opm' or 'squid'].")
def generate_qc_report(file, outdir, data_type):
    """
    Generate Quality Control Report of MEG.

    Parameters
    ----------
    file : string
        the name of meg file required for quality assessment.
    outdir : str
        the output path of quality report.
    data_type: str
        the type of MEG data.['opm' or 'squid']
    Returns
    -------
        Reports
    """
    filename = Path(file).stem+'.report'
    gen_quality_report([file], outdir=outdir, report_fname=filename, data_type=data_type, ftype='html')

