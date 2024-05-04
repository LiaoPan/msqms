# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :
   Author :       LiaoPan
-------------------------------------------------
   Change Activity:
                   2023/10/16:
-------------------------------------------------
"""
__author__ = 'LiaoPan'

import click
import os.path as op

from mne.io import read_raw_fif
from opmqc.main import test_squid_fif_path, test_opm_fif_path
from opmqc.reports.report import gen_quality_report

# config click
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--file', type=str, required=True, help='the fif file required for quality assessment.')
@click.option('--outdir', type=str, required=True, default='.', show_default=True, help='the output path of quality report.')
def generate_qc_report(file, outdir):
    """
    Generate Quality Control Report of OPM-MEG.
    """
    gen_quality_report([file], outdir=op.join(outdir, "report.html"))


def build_qc_workflow(config_file):
    """
    Create the Nipype workflow.

    Parameters
    ----------
    config_file

    Returns
    -------

    """
    pass
