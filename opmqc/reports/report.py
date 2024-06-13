# -*- coding: utf-8 -*-
"""Generate MEG Pipeline HTML Report"""
import mne
import json
import jinja2
import os.path as op
from box import Box

from tqdm.auto import tqdm
from typing import Union
from pathlib import Path
from jinja2 import Environment, PackageLoader
from mne.io import read_raw_fif

from opmqc.qc import get_header_info
from opmqc.utils.logging import clogger
from opmqc.qc.msqm import MSQM
from opmqc.constants import METRICS_COLUMNS, METRICS_REPORT_MAPPING, METRICS_MAPPING


def gen_quality_report(megfiles: [Union[str, Path]], outdir: Union[str, Path], ftype: str = 'html'):
    """Generate HTML/JSON Report for a set of MEG Raw data.

    Parameters
    ----------
    megfiles : [Union[str, Path]]
    outdir : Union[str, Path]
        the folder where the report will be saved.
    ftype : str
        the type of generated report file.
    Returns
    -------
    """
    # validate meg files.

    # validate outdir

    for fmeg in megfiles:
        clogger.info(f"Generating report for {fmeg}")

        # check meg file
        if not op.exists(fmeg):
            clogger.error(f"{fmeg} is not exists. Please check the path of file.")
        raw = read_raw_fif(fmeg, verbose=False, preload=True)

        # compute the msqm score and obtain the reference values & hints[↑↓✔]
        # For examples:
        # "msqm_score":98,
        # "S": {"lower_bound","upper_bound,"hints":"✔"}
        # "I": {"score":0.9,"value":10e-12,"lower_bound":,"upper_bound,"hints":"↓"}

        raw.filter(0.1, 100, n_jobs=-1, verbose=False).notch_filter([50, 100], verbose=False, n_jobs=-1)
        msqm = MSQM(raw, 'opm', verbose=10, n_jobs=4)
        msqm = msqm.compute_msqm_score()
        msqm_score = msqm['msqm_score']
        details = msqm['details']
        category_scores = msqm['category_scores']

        # for debug
        # msqm_score = 0.98
        # details = {}
        # category_scores = {"time_domain": 0.3, 'artifacts': 0.2, 'frequency_domain': 0.2, 'entropy': 0.2,
        #                    'fractal': 0.2}

        info = get_header_info(raw)
        qreport = QualityReport(report_data=Box(
            {"Overview": info,
             "Quality_Ref": {"msqm_score": msqm_score, "details": details, "category_scores": category_scores}}),
            minify_html=False)

        if ftype == "json":
            qreport.to_json(outdir)
        else:
            qreport.to_html(outdir)


class QualityReport(object):
    """
    Generate a quality report from MEG raw data.
    """

    def __init__(self,
                 report_data,
                 minify_html,
                 ):
        self.report_data = report_data
        self.minify_html = minify_html

    def to_json(self, out_json_path: Union[str, Path]) -> None:
        """
        write the report to json file
        """
        with tqdm(total=1, desc="Render JSON") as pbar:
            report_data = json.dumps(self.report_data, indent=4)
            pbar.update()
        self._to_file(report_data=report_data, output_file=out_json_path)

    def to_html(self, out_html_path: Union[str, Path]) -> None:
        """
        write the report to html file
        """
        with tqdm(total=1, desc="Render Html") as pbar:
            html = HtmlReport(self.report_data).render_html()

            if self.minify_html:
                import minify_html
                html = minify_html.minify(html,
                                          minify_js=True,
                                          minify_css=True,
                                          )
            pbar.update()

        self._to_file(report_data=html, output_file=out_html_path)

    def _to_file(self, report_data: str, output_file: Union[str, Path]) -> None:
        """
        Write the report to a file.
        """
        if not isinstance(output_file, Path):
            output_file = Path(str(output_file))

        if output_file.suffix not in [".html", ".json"]:
            suffix = output_file.suffix
            output_file = output_file.with_suffix(".html")
            clogger.warning(
                f"Extension {suffix} not supported. We use .html instead."
                f"To remove this warning, please use .html or .json."
            )

        with tqdm(total=1, desc="Export quality report to file") as pbar:
            output_file.write_text(report_data, encoding="utf-8")
            pbar.update()
        clogger.info(f"Export quality report path:{output_file}")


class HtmlReport(object):
    def __init__(self, report_data):
        # Init Jinja
        package_loader = PackageLoader(package_name="opmqc", package_path="reports/templates")
        self.jinja2_env = Environment(loader=package_loader)

        self.report_data = report_data

        self.nav_title = "<strong>MEG Quality Report</strong>"
        self.info_title_name = "MEG Data Info"
        self.overview_title_name = "MEG Quality Overview"

        # MEG data info
        self.info_tabs = [("Overview", "overview"), ("Participant Info", "participantinfo"), ("MEG Info", "meginfo")]

        # basic info
        basic_info = self.report_data.Overview.basic_info
        meg_info = self.report_data.Overview.meg_info
        self.info_basic = {
            "Manufacturer": basic_info.Experimenter,
            "Duration": basic_info.Duration,
            "Frequency": basic_info.Sampling_frequency,
            "Highpass": basic_info.Highpass,
            "Lowpass": basic_info.Lowpass,
            "Data Size": basic_info.Data_size,
            "Bad Channels": basic_info.Bad_channels,
            "Measurement date": basic_info.Measurement_date,
            "Source filename": basic_info.Source_filename,
        }

        # basic participant info
        self.info_participant_dict = {
            "Name": basic_info.Participant.name,
            "Birthday": basic_info.Participant.birthday,
            "Gender": basic_info.Participant.sex,
        }
        # basic meg info
        self.info_meg_list = [("Channel Type", "Value"),
                              ("Mag", meg_info.n_mag),
                              ("Grad", meg_info.n_grad),
                              ("Stim", meg_info.n_stim),
                              ("EEG", meg_info.n_eeg),
                              ("ECG", meg_info.n_ecg),
                              ("EOG", meg_info.n_eog),
                              ("Digitized points", meg_info.n_dig)
                              ]
        # quality reference
        quality_ref_details = self.report_data.Quality_Ref.details
        quality_ref_overview = self.report_data.Quality_Ref.category_scores

        # msqm score
        self.msqm_score = self._format_msqm_score(self.report_data.Quality_Ref.msqm_score)
        clogger.info("MSQM score: {}".format(self.msqm_score))
        #format css for msqm score
        self.msqm_score_css = self._css_style_for_msqm_score()

        # overview of category metrics
        self.overview_quality_list = [
            ("Quality Indices", "Value", "Ref Value", "Status"),
        ]

        for k, v in quality_ref_overview.items():
            hint = self._get_hint(v)
            self.overview_quality_list.append((METRICS_REPORT_MAPPING[k], f"{v:.3f}", [0, 1], hint))

        # time domain, frequency domain etc.
        html_table_col_names = ("Quality Indices", "Quality Score", "Value", "Ref Value", "Status")
        metric_cate_name = quality_ref_overview.keys()

        for cate_name in metric_cate_name:
            metric_column_names = METRICS_COLUMNS[cate_name]
            quality_list = [html_table_col_names]
            try:
                for m_cn in metric_column_names:
                    m_scores = quality_ref_details[m_cn]
                    quality_score = m_scores['quality_score']
                    value = m_scores['metric_score']
                    lower_bound = m_scores['lower_bound']
                    upper_bound = m_scores['upper_bound']
                    hint = m_scores['hint']

                    # replace metric name in html report.
                    if m_cn in METRICS_MAPPING.keys():
                        m_cn = METRICS_MAPPING[m_cn]

                    value = self._format_number(value)
                    lower_bound = self._format_number(lower_bound)
                    upper_bound = self._format_number(upper_bound)
                    content = (m_cn, f"{quality_score:.3f}", value, [lower_bound, upper_bound], hint)
                    quality_list.append(content)
            except Exception as e:
                clogger.error(e)

            if cate_name == "time_domain":
                self.time_quality_list = quality_list
            elif cate_name == "frequency_domain":
                self.freq_quality_list = quality_list
            elif cate_name == "entropy":
                self.entropy_quality_list = quality_list
            elif cate_name == "fractal":
                self.fractal_quality_list = quality_list
            elif cate_name == "artifacts":
                self.artifacts_quality_list = quality_list

        self.overview_tabs = [("Overview", "overview"),
                              ("Time Series", "timeseries"),
                              ("Frequencies", "frequencies"),
                              ("Entropy", "entropy"),
                              ("Fractal", "fractal"),
                              ("Artifacts", "artifacts")]
        # self.overview_tabs = ["view", "info", "meg"]

        self.overview_quality_dict = {

        }

        self.footer = 'MEG Quality Report Generated by <a href="https://github.com/liaopan/opmqc">OPMQC</a>.'

    @staticmethod
    def _format_msqm_score(score):
        score = score * 100
        if score < 40:
            return f"{score:.2f}/Bad"
        elif score >= 40 and score < 60:
            return f"{score:.2f}/Poor"
        elif score >= 60 and score < 80:
            return f"{score:.2f}/Fair"
        elif score >= 80 and score < 90:
            return f"{score:.2f}/Good"
        elif score >= 90:
            return f"{score:.2f}/Excellent"

    @staticmethod
    def _get_hint(score):
        if score < 0.6:
            return "Low"
        elif score >= 0.6 and score < 0.8:
            return "Medium"
        elif score >= 0.8 and score <= 1:
            return "High"

    @staticmethod
    def _format_number(value, threshold=1e3):
        """
        Format the number based on its size.

        If the absolute value is larger than the threshold or smaller than 1/threshold,
        the number is formatted using scientific notation. Otherwise, it retains three decimal places.

        Parameters
        ----------
        value : float
            The number to be formatted.
        threshold : float, optional
            The threshold to decide whether to use scientific notation. Default is 1e6.

        Returns
        -------
        str
            The formatted string.
        """
        if abs(value) >= threshold or abs(value) < 1 / threshold:
            return f"{value:.2e}"  # Use scientific notation
        else:
            return f"{value:.3f}"  # Keep three decimal places

    def _css_style_for_msqm_score(self):
        """Change the css style depending on the msqm score.
        """
        color = None
        excellent_color = '#4fb332'
        good_color = '#88cecf'
        fair_color = '#2ab9cd'
        poor_color = '#f5af30'
        bad_color = '#e52f10'
        score = float(self.msqm_score.split("/")[0])

        if "Bad" in self.msqm_score:
            color = bad_color
        elif "Poor" in self.msqm_score:
            color = poor_color
        elif "Fair" in self.msqm_score:
            color = fair_color
        elif "Good" in self.msqm_score:
            color = good_color
        elif "Excellent" in self.msqm_score:
            color = excellent_color

        if color is not None:
            style = f"--c:{color};--p:{score:.2f}"
        else:
            style = ""

        return style

    def get_template(self, template_name: str) -> jinja2.Template:
        return self.jinja2_env.get_template(template_name)

    def gen_base_template(self):
        return self.get_template('base.html')

    def gen_html_report(self):
        # navigation settings.
        self.nav_items = [("INFO", "info"),
                          ("Quality Overview", "overview"),
                          # ("Artifacts", "artifacts"),
                          ("Visual Inspection", "inspection"),
                          # ("ICA", "ica"),
                          ]

        # get base templates(Main HTML)
        render_params = {
            "title": self.nav_title,
            "nav": True,
            "nav_items": self.nav_items,
            "footer": self.footer,

            "info_title": self.info_title_name,
            "info_tabs": self.info_tabs,

            "info_basic": self.info_basic,
            "info_participant_dict": self.info_participant_dict,
            "info_meg_list": self.info_meg_list,

            "msqm_score": self.msqm_score,
            "overview_title": self.overview_title_name,
            "overview_tabs": self.overview_tabs,
            "overview_quality_list": self.overview_quality_list,
            "artifacts_quality_list": self.artifacts_quality_list,
            "time_quality_list": self.time_quality_list,
            "freq_quality_list": self.freq_quality_list,
            "entropy_quality_list": self.entropy_quality_list,
            "fractal_quality_list": self.fractal_quality_list,

            "msqm_score_css": self.msqm_score_css,
            # "overview_dict": self.overview_dict,
        }

        html = self.gen_base_template().render(**render_params)
        return html

    def gen_nav_html(self):
        html = self.get_template("navigation.html")
        # multi panels
        self.nav_items = [("Quality Overview", "overview"),
                          ("Artifacts", "artifacts"),
                          ("Visual Inspection", "inspection"),
                          ("ICA", "ica")]

        html.render(self.nav_items)
        return html

    def gen_body_html(self):
        html = self.get_template("navigation.html")
        # multi panels
        self.nav_items = [("Quality Overview", "overview"),
                          ("Artifacts", "artifacts"),
                          ("Visual Inspection", "inspection"),
                          ("ICA", "ica")]

        html.render(self.nav_items)
        # panel: Quality Overview

        # panel: Artifacts

        # panel: Visual Inpsection

        # panel: ICA(optional)

        return html

    def render_html(self):
        html_page = self.gen_html_report()
        # print(html_page)
        return html_page


if __name__ == "__main__":
    navigation_title = "MEG Quality Report"
    navigation_links = ["Quality Overview", "Artifacts", "Quality Visual Inspection", "ICA"]
    from opmqc.main import test_opm_fif_path, test_squid_fif_path

    # gen_quality_report(["/Volumes/Touch/Code/osl_practice/anonymize_raw_tsss.fif"], outdir="./demo_report.html")
    gen_quality_report([test_opm_fif_path], outdir="./new_demo_report.html")
