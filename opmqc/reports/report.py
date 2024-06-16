# -*- coding: utf-8 -*-
"""Generate MEG Pipeline HTML Report"""
import os.path

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
from opmqc.constants import DATA_TYPE
from opmqc.qc.visual_inspection import VisualInspection
from opmqc.constants import METRICS_COLUMNS, METRICS_REPORT_MAPPING, METRICS_MAPPING


def gen_quality_report(megfiles: [Union[str, Path]], outdir: Union[str, Path],report_fname:str="", data_type: DATA_TYPE="",ftype: str = 'html'):
    """Generate HTML/JSON Report for a set of MEG Raw data.

    Parameters
    ----------
    megfiles : [Union[str, Path]]
    outdir : Union[str, Path]
        the folder where the report will be saved.
    report_fname: str
        the name of report.
    data_type: str
        the type of data.['opm' or 'squid']
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

        raw_filter = raw.copy().filter(0.1, 100, n_jobs=-1, verbose=False).notch_filter([50, 100], verbose=False, n_jobs=-1)
        msqm = MSQM(raw_filter, origin_raw=raw, data_type=data_type, verbose=10, n_jobs=4)
        msqm_dict = msqm.compute_msqm_score()
        msqm_score = msqm_dict['msqm_score']
        details = msqm_dict['details']
        category_scores = msqm_dict['category_scores']

        fmeg_fname = Path(raw.filenames[0]).stem
        vis = VisualInspection(raw=raw, output_fpath=os.path.join(outdir, f'{fmeg_fname}.imgs'))
        meg_data = raw.get_data('mag')

        nan_mask = msqm.nan_mask
        bad_chan_mask = msqm.bad_chan_mask
        bad_seg_mask = msqm.bad_seg_mask
        flat_mask = msqm.flat_mask
        bad_chan_names = msqm.bad_chan_names
        zero_mask = msqm.zero_mask

        vis.visual_psd()
        vis.visualize_heatmap(data=meg_data, bad_mask=zero_mask, filename="Heatmap_zerovalue.html",
                              label='ZeroValue')
        vis.visualize_heatmap(data=meg_data, bad_mask=nan_mask, filename="Heatmap_NaN.html",
                              label='NaN')
        vis.visualize_heatmap(data=meg_data, bad_mask=bad_chan_mask, filename="Heatmap_bad_channels.html",
                              label='BadChannel')
        vis.visualize_heatmap(data=meg_data, bad_mask=bad_seg_mask, filename="Heatmap_bad_segments.html",
                              label='BadSegments')
        vis.visualize_heatmap(data=meg_data, bad_mask=flat_mask, filename="Heatmap_flat_channels.html",
                              label='Flat')
        vis.visual_bad_channel_topomap(bad_channels=bad_chan_names, filename="Bad_channels_distribution.png",
                                       show_names=True)

        # for debug
        # msqm_score = 0.88
        # details = {'max_ptp': {'quality_score': 1, 'metric_score': 4.210898388610023e-12, 'lower_bound': -2.3602284324990543e-07, 'upper_bound': 2.6493921972267915e-07, 'hint': '✔'}, 'S': {'quality_score': 0.636571456992661, 'metric_score': 1.9430031627543123, 'lower_bound': 2.623627306847128, 'upper_bound': 4.4964140133086286, 'hint': '↓'}, 'C': {'quality_score': 0.8390398792210456, 'metric_score': 10.48615742957991, 'lower_bound': 11.369409015417698, 'upper_bound': 16.85680292120852, 'hint': '↓'}, 'I': {'quality_score': 0.6980531016926521, 'metric_score': 20.329387368006188, 'lower_bound': 31.923511813784586, 'upper_bound': 70.32140399539887, 'hint': '↓'}, 'L': {'quality_score': 1, 'metric_score': 1.7146758299291877e-05, 'lower_bound': -0.0047155314087559555, 'upper_bound': 0.007008033308854388, 'hint': '✔'}, 'mmr': {'quality_score': 1, 'metric_score': 9.688842656372957e-12, 'lower_bound': -7.945065272613537e-07, 'upper_bound': 9.563664839987926e-07, 'hint': '✔'}, 'max_field_change': {'quality_score': 1, 'metric_score': 8.898472110522491e-13, 'lower_bound': -1.3707800756858704e-07, 'upper_bound': 1.6579044812575712e-07, 'hint': '✔'}, 'mean_field_change': {'quality_score': 1, 'metric_score': 2.1588871514957103e-14, 'lower_bound': -1.1223198749742977e-09, 'upper_bound': 1.2619573968340738e-09, 'hint': '✔'}, 'std_field_change': {'quality_score': 1, 'metric_score': 1.9845503045623648e-14, 'lower_bound': -1.984192939785529e-09, 'upper_bound': 2.3121304240874602e-09, 'hint': '✔'}, 'rms': {'quality_score': 1, 'metric_score': 9.233427346002339e-13, 'lower_bound': -5.047210123157186e-08, 'upper_bound': 6.175431918543353e-08, 'hint': '✔'}, 'arv': {'quality_score': 1, 'metric_score': 4.8084900650295e-13, 'lower_bound': -1.58680313745648e-08, 'upper_bound': 1.868690817597984e-08, 'hint': '✔'}, 'mean': {'quality_score': 1, 'metric_score': 1.244763710138051e-16, 'lower_bound': -4.171571251074098e-09, 'upper_bound': 5.321189796997719e-09, 'hint': '✔'}, 'variance': {'quality_score': 1, 'metric_score': 8.537696635943241e-25, 'lower_bound': -1.4282025550921224e-13, 'upper_bound': 1.64282023562554e-13, 'hint': '✔'}, 'std_values': {'quality_score': 1, 'metric_score': 9.233406553783618e-13, 'lower_bound': -5.016812603103612e-08, 'upper_bound': 6.136265224339484e-08, 'hint': '✔'}, 'max_values': {'quality_score': 1, 'metric_score': 4.864015795668343e-12, 'lower_bound': -6.063888006058493e-07, 'upper_bound': 7.338068429236114e-07, 'hint': '✔'}, 'min_values': {'quality_score': 1, 'metric_score': -4.824826860704613e-12, 'lower_bound': -2.3952317312808677e-07, 'upper_bound': 2.0508125870840996e-07, 'hint': '✔'}, 'median_values': {'quality_score': 1, 'metric_score': -1.4347550864406708e-15, 'lower_bound': -2.405120604068551e-10, 'upper_bound': 2.557771831620386e-10, 'hint': '✔'}, 'hjorth_mobility': {'quality_score': 0, 'metric_score': 0.03166494337708053, 'lower_bound': 0.01013033613615232, 'upper_bound': 0.02416418566031306, 'hint': '✘'}, 'hjorth_complexity': {'quality_score': 0, 'metric_score': 6.350346580480347, 'lower_bound': 22.890677614955045, 'upper_bound': 36.8436539670498, 'hint': '✘'}, 'num_of_zero_crossings': {'quality_score': 0, 'metric_score': 0.021937002852193698, 'lower_bound': 0.003070693032658337, 'upper_bound': 0.021341148222957174, 'hint': '✘'}, 'DFA': {'quality_score': 0.3391594254634005, 'metric_score': 1.2738102058565535, 'lower_bound': 1.3511475941139013, 'upper_bound': 1.468176407472674, 'hint': '↓'}, 'max_mean_offset': {'quality_score': 0.05239873874132572, 'metric_score': 5.3753023025987705e-15, 'lower_bound': 2.349895905397193e-08, 'upper_bound': 4.8297315324051794e-08, 'hint': '↓'}, 'mean_offset': {'quality_score': 0, 'metric_score': 1.5658019607978766e-15, 'lower_bound': 7.625835425053466e-10, 'upper_bound': 1.5114204288677438e-09, 'hint': '✘'}, 'Zero_ratio': {'quality_score': 0, 'metric_score': 2.991443019976258e-05, 'lower_bound': 1.0442435968838144e-06, 'upper_bound': 2.5057697378509674e-06, 'hint': '✘'}, 'std_mean_offset': {'quality_score': 0.0365161833706501, 'metric_score': 1.1682583555008518e-15, 'lower_bound': 2.882874800947118e-09, 'upper_bound': 5.875009783322018e-09, 'hint': '↓'}, 'max_median_offset': {'quality_score': 1, 'metric_score': 3.830726374525228e-14, 'lower_bound': -1.4891710347261785e-09, 'upper_bound': 3.0224937251558155e-09, 'hint': '✔'}, 'median_offset': {'quality_score': 1, 'metric_score': 1.410822042046943e-14, 'lower_bound': -1.8422225583199954e-11, 'upper_bound': 3.801576774448462e-11, 'hint': '✔'}, 'std_median_offset': {'quality_score': 1, 'metric_score': 9.191078203473446e-15, 'lower_bound': -1.6531834976119388e-10, 'upper_bound': 3.361236437270235e-10, 'hint': '✔'}, 'p1': {'quality_score': 1, 'metric_score': 2.4832067047934644e-16, 'lower_bound': -9.591375683090533e-12, 'upper_bound': 1.1301554434520452e-11, 'hint': '✔'}, 'p2': {'quality_score': 1, 'metric_score': 1.667272495104182e-15, 'lower_bound': -8.395036169580603e-11, 'upper_bound': 1.0301607143740373e-10, 'hint': '✔'}, 'p3': {'quality_score': 0, 'metric_score': 13.051892184840463, 'lower_bound': 55.77079859170577, 'upper_bound': 94.04879306196314, 'hint': '✘'}, 'p4': {'quality_score': 0.8292491605732685, 'metric_score': 217.68969334431858, 'lower_bound': 2305.05207596116, 'upper_bound': 14529.661861920777, 'hint': '↓'}, 'p5': {'quality_score': 0, 'metric_score': 11.53772276948559, 'lower_bound': 22.547078527942592, 'upper_bound': 29.888380741483562, 'hint': '✘'}, 'p6': {'quality_score': 1, 'metric_score': 2.7658187426768086e-07, 'lower_bound': -1.5768496151932156e-05, 'upper_bound': 2.530029412795343e-05, 'hint': '✔'}, 'p7': {'quality_score': 0, 'metric_score': 21.064549884092635, 'lower_bound': 47.136415086333095, 'upper_bound': 61.943142022865636, 'hint': '✘'}, 'p8': {'quality_score': 0, 'metric_score': 103.55648198443087, 'lower_bound': 208.34594301408768, 'upper_bound': 263.66065100566533, 'hint': '✘'}, 'p9': {'quality_score': 0.9611573509700524, 'metric_score': 0.2087925817155825, 'lower_bound': 0.21063572116970164, 'upper_bound': 0.258087156644612, 'hint': '↓'}, 'p10': {'quality_score': 1, 'metric_score': 2.4007934211805027e-08, 'lower_bound': -7.464103895573939e-07, 'upper_bound': 1.2350452365259103e-06, 'hint': '✔'}, 'p11': {'quality_score': 0, 'metric_score': 246503107.92904365, 'lower_bound': 69291243.81429276, 'upper_bound': 164658062.69994727, 'hint': '✘'}, 'p12': {'quality_score': 0, 'metric_score': 1.5377136730303622e+17, 'lower_bound': 7806190304380896.0, 'upper_bound': 7.057538000383507e+16, 'hint': '✘'}, 'p13': {'quality_score': 1, 'metric_score': 5.408849834352705e-12, 'lower_bound': -8.739443023589972e-09, 'upper_bound': 1.068327565061985e-08, 'hint': '✔'}, 'permutation_entropy': {'quality_score': 0, 'metric_score': 0.5166734544214481, 'lower_bound': 0.6641767444467576, 'upper_bound': 0.7346060768086465, 'hint': '✘'}, 'spectral_entropy': {'quality_score': 0.4380497715833509, 'metric_score': 0.27676763057922776, 'lower_bound': 0.3449191998212739, 'upper_bound': 0.4661960777292516, 'hint': '↓'}, 'svd_entropy': {'quality_score': 0, 'metric_score': 0.1156067172245473, 'lower_bound': 0.05112847417811117, 'upper_bound': 0.09246998490193437, 'hint': '✘'}, 'approximate_entropy': {'quality_score': 0, 'metric_score': 0.09506681629683165, 'lower_bound': -0.002183724451159292, 'upper_bound': 0.057771169014295964, 'hint': '✘'}, 'sample_entropy': {'quality_score': 0, 'metric_score': 0.078396306543074, 'lower_bound': -0.0016331515242551733, 'upper_bound': 0.05194606203916407, 'hint': '✘'}, 'power_spectral_entropy': {'quality_score': 0.43804977158335046, 'metric_score': 1.34503876421472, 'lower_bound': 1.6762426057939293, 'upper_bound': 2.265625481413375, 'hint': '↓'}, 'Total_Energy': {'quality_score': 1, 'metric_score': 8.554415163699364e-20, 'lower_bound': -5.449294969381888e-08, 'upper_bound': 6.265338795968155e-08, 'hint': '✔'}, 'Total_Entropy': {'quality_score': 0, 'metric_score': 35.68322802369352, 'lower_bound': 17.650418174617535, 'upper_bound': 34.229796349908575, 'hint': '✘'}, 'Energy_Entropy_Ratio': {'quality_score': 1, 'metric_score': 2.4016681733929615e-21, 'lower_bound': -1.397143406697829e-09, 'upper_bound': 1.6339349602116215e-09, 'hint': '✔'}, 'PFD': {'quality_score': 0.6140837342674625, 'metric_score': 1.0017195951496918, 'lower_bound': 1.0030936843330922, 'upper_bound': 1.0066542732623827, 'hint': '↓'}, 'KFD': {'quality_score': 0, 'metric_score': 2.100654219989885, 'lower_bound': 1.456704486133142, 'upper_bound': 1.8462845606170841, 'hint': '✘'}, 'HFD': {'quality_score': 0, 'metric_score': 1.039029700811997, 'lower_bound': 1.237477492170722, 'upper_bound': 1.3493118670462148, 'hint': '✘'}, 'BadChanRatio': {'quality_score': 0, 'metric_score': 0.11538461538461539, 'lower_bound': 0.0, 'upper_bound': 0.04, 'hint': '✘'}, 'BadSegmentsRatio': {'quality_score': 1, 'metric_score': 0.0024937655860348684, 'lower_bound': 0.0, 'upper_bound': 0.0025, 'hint': '✔'}, 'NaN_ratio': {'quality_score': 1, 'metric_score': 0.0, 'lower_bound': 0.0, 'upper_bound': 0.0025, 'hint': '✔'}, 'Flat_chan_ratio': {'quality_score': 0, 'metric_score': 97.43589743589743, 'lower_bound': 0.0, 'upper_bound': 0.0025, 'hint': '✘'}}
        # category_scores = {"time_domain": 0.3, 'artifacts': 0.2, 'frequency_domain': 0.2, 'entropy': 0.2,
        #                    'fractal': 0.2}

        info = get_header_info(raw)
        qreport = QualityReport(report_data=Box(
            {"Overview": info,
             "Quality_Ref": {"msqm_score": msqm_score, "details": details, "category_scores": category_scores}}),
            minify_html=False)

        report_name = os.path.join(outdir,f"{report_fname}.{ftype}")
        if ftype == "json":
            qreport.to_json(report_name)
        else:
            qreport.to_html(report_name)


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
                    content = (m_cn, f"{quality_score:.3f}", value, f"[{lower_bound}, {upper_bound}]", hint)
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
            "report_html_name": Path(self.info_basic["Source filename"]).stem,
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
    gen_quality_report([test_squid_fif_path], outdir=r"C:\Data\Code\opmqc\opmqc\reports",data_type='squid',report_fname="new_demo_report",ftype='html')
    # gen_quality_report([test_opm_fif_path], outdir=r"C:\Data\Code\opmqc\opmqc\reports",data_type='opm',report_fname="new_demo_report",ftype='html')
