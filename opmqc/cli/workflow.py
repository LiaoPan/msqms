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

from opmqc.reports.report import gen_quality_report

# config click
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--file', '-f', type=str, required=True, help='the fif file required for quality assessment.')
@click.option('--outdir', '-o', type=str, required=True, default='.', show_default=True, help='the output path of quality report.')
def generate_qc_report(file, outdir):
    """
    Generate Quality Control Report of OPM-MEG.
    """
    gen_quality_report([file], outdir=outdir, report_fname="report",ftype='html')


def build_qc_workflow(config_file):
    """
    Create the Nipype workflow.

    Parameters
    ----------
    config_file

    Returns
    -------

    """
    raise NotImplementedError
