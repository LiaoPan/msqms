# -*- coding: utf-8 -*-
"""
Example: Generate Summary Quality Report for Multiple MEG Files

This example demonstrates how to generate a summary quality report
that aggregates quality scores from multiple MEG files and provides
visualizations and easy navigation between individual reports.
"""

from pathlib import Path
from msqms.reports import gen_summary_quality_report

# Example: Generate summary report for multiple MEG files
if __name__ == "__main__":
    # List of MEG file paths
    meg_files = [
        "path/to/file1.fif",
        "path/to/file2.fif",
        "path/to/file3.fif",
        # ... add more files as needed
    ]
    
    # Output directory for reports
    output_dir = "quality_reports"
    
    # Data type: 'opm' or 'squid'
    data_type = "opm"
    
    # Generate summary report
    summary_stats = gen_summary_quality_report(
        megfiles=meg_files,
        outdir=output_dir,
        report_fname="summary_report",
        data_type=data_type,
        ftype='html'
    )
    
    if summary_stats:
        print(f"Summary report generated successfully!")
        print(f"Total files processed: {summary_stats['statistics']['total_files']}")
        print(f"Average MSQM score: {summary_stats['statistics']['msqm_mean']:.3f}")
        print(f"Summary report saved to: {output_dir}/summary_report.html")
    else:
        print("Failed to generate summary report.")

