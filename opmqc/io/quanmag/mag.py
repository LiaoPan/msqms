# -*- coding: utf-8 -*-
"""Load Quanmag@company OPM-MEG data, and convert Quan-Mag opm data(*.mag) to other file(*.fif or *.mat)"""


import os
import sys
import mne
import struct
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from os.path import exists
from datetime import datetime, timezone
from collections import defaultdict
from mne.io.constants import FIFF


from opmqc.utils.utils import fill_zeros_with_nearest_value
from opmqc.utils.logging import clogger

def _mag2data(mag_path, check_rf_increment=True, check_ch_zero=True, interpolate_ch_nearest=True, verbose=False):
    """
    convert mag file(*.mag) to dict[numpy object].

    opm data(*.mag):
    Endianness: Little-Endian
    [1] Header:software name, software version,patient id, patient name,sample rate, filter info, sensors info ,etc.
    [2] Block/Section: Sensor Data and Trigger Data.
    [3] Checkout

    sfreq: default:1000Hz, 1ms

    Parameters:
      mag_path : str
        the path of quanmag opm-meg data(*.mag).
      verbose : bool
        True: clogger.info debug-log info; False: don't clogger.info ang info.

    Returns:
      -  Return Numpy Ndarray Object with event trigger(STI101)

    Ref:
      - https://docs.python.org/zh-cn/3/library/struct.html

    """
    # define inner functions
    log = lambda key, value: clogger.info("{}:{}".format(key, value)) if verbose else None

    #  check opm file.
    if not exists(mag_path):
        clogger.info("opm path is not exists!")

    # patient info
    patient_info = {}  # store patient info

    # sensor info: channel data and rf pulse cnt
    channel_datas = []  # store sensor data
    channel_rf_cnts = []  # store rf pulse cnt

    # trigger info: store trigger data
    trigger_values = []
    trigger_rf_cnts = []

    try:
        with open(mag_path, 'rb') as fid:
            # drop bytes.
            fid.read(16)

            clogger.info("STEP 0.Parse Patient Info")
            # 1.file header
            header = fid.read(8).decode('utf-8', errors='ignore')  # char
            patient_info["Header"] = header.replace('\x00', '')
            log('1.Header', header)

            # 2.software name
            software_name = fid.read(32).decode('utf-8', errors='ignore')  # char
            patient_info["software_name"] = software_name.replace('\x00', '')
            log("2.Software_name", software_name)

            # 3.software version
            software_version = struct.unpack('<Q', fid.read(8))[0]  # unit64
            patient_info["software_version"] = software_version
            log("3.Software_version", software_version)

            # 4.patient id
            patient_id = fid.read(32).decode('utf-8', errors='ignore')
            patient_info["patient_id"] = patient_id.replace('\x00', '')
            log('4.Patient_id', patient_id)

            # 5. 6. patient name: patient first name and last name.
            patient_name = fid.read(64).decode('utf-8', errors='ignore')
            patient_info["patient_name"] = patient_name.replace('\x00', '')
            log('5.Patient_name', patient_name)

            # 7.patient birthday
            patient_birthday = fid.read(20).decode('utf-8', errors='ignore')
            patient_info["patient_birthday"] = patient_birthday.replace('\x00', '')
            log("6.Patient_birthday", patient_birthday)

            # 8. patient gender
            patient_gender = _get_gender(fid.read(1))  # check 1
            patient_info["patient_gender"] = patient_gender
            log("7.Patient_gender", patient_gender)

            # 9.examination project name
            exam_name = fid.read(32).decode('utf-8', errors='ignore')
            patient_info["exam_name"] = exam_name.replace('\x00', '')
            log("8.Exam_name", exam_name)

            # 10.examination time
            exam_time = fid.read(20).decode('utf-8', errors='ignore')
            patient_info["exam_time"] = exam_time.replace('\x00', '')
            log("9.Exam_time", exam_time)

            # 11.record id
            record_id = struct.unpack('<I', fid.read(4))[0]
            patient_info["record_id"] = record_id
            log("10.Record_ID", record_id)

            # 12.number of channel
            nchan = struct.unpack('<I', fid.read(4))[0]
            patient_info["nchan"] = nchan
            log("11.NumberOfChannel", nchan)

            # 13.firmware version
            firmware_ver = struct.unpack('<Q', fid.read(8))[0]
            patient_info["firmware_ver"] = firmware_ver
            log("12.Firmware", firmware_ver)

            # 14.sample rate
            sample_rate = struct.unpack('<i', fid.read(4))[0]
            patient_info["sample_rate"] = sample_rate / 2
            log("13.Sample_rate", sample_rate / 2)

            # 15.type of filter
            filter_type = _get_filter_type(fid.read(1))
            patient_info["filter_type"] = filter_type
            log("14.Filter_type", filter_type)

            # 16.number of Block
            nblock = struct.unpack('<Q', fid.read(8))[0]
            log("16.NumberOfBlock", nblock)

            clogger.info(f"Patient Info:{patient_info}")
            # struct:MagChannelFilter: 17 bytes (16 bytes + 1 byte)
            # 512 * sizeof(MagChannelFilter)
            mag_channel_filter = struct.unpack('<' + 'ddb' * 512, fid.read(17 * 512))
            # log("17.MagChannelFilter",mag_channel_filter)

            # ---- Parse Sensor Data Blocks ----
            clogger.info("STEP 1.Parse Sensor Data Blocks")

            ## block header: the number of section. | 8 bytes
            num_sections = struct.unpack('<Q', fid.read(8))[0]  # 64
            log("Num_Sections", num_sections)

            # Handle Channel Data.

            for sec_id in tqdm(range(1, num_sections + 1, 1), file=sys.stdout):
                print(f" Parsing channel: {sec_id}", end='', flush=True)
                ## section header: the number of datas.
                ##  517920 [data line number] * 56 [sensor_data:channel(8 bytes),rf pulse(8 bytes),sensor_value(4*10 bytes)] = 29003520
                # The number of bytes of the channel | 8 bytes
                num_ch_bytes = struct.unpack('<Q', fid.read(8))[0]
                num_ch_line = int(num_ch_bytes / 56)  # 10 lines of data are decoded once:  51792 * 10
                log("Num_Ch_Bytes", num_ch_bytes)
                log("Num_Ch_Line", num_ch_line)

                ## decoding channel data
                single_ch_data = []
                single_rf_cnt = []
                for i in range(num_ch_line):
                    ## sensor data definition
                    # channel.id? | 8 bytes
                    ch_id = struct.unpack('<Q', fid.read(8))[0]
                    log("Ch_Id?", ch_id)

                    # sync pulse counting| 8 bytes
                    sync_rf_count = struct.unpack('<2I', fid.read(8))[1]
                    log("Sync_RF_Count:", sync_rf_count)

                    # data*.txt
                    ch_data = struct.unpack('<10f', fid.read(4 * 10))  # 10 lines of data are decoded once
                    log("Channel_data:", ch_data)

                    # append sensor and rf pulse data: 10 line data added once.
                    ch_data = list(ch_data)
                    rf_cnt = range(sync_rf_count, sync_rf_count + 20, 2)  # rf pulse cnt incremented by 2;
                    if len(ch_data) != len(rf_cnt):
                        warnings.warn("The size of channel data and the pulse signal are inconsistent!")
                    single_ch_data.extend(ch_data)
                    single_rf_cnt.extend(rf_cnt)

                single_rf_cnt = np.ceil(np.array(single_rf_cnt) / 2)
                # check increment(+1)
                if check_rf_increment:
                    index = np.where(np.all(np.diff(single_rf_cnt) == 1) == False)[0]
                    if len(index) == 1:
                        warnings.warn(f"The RF pulse is not incremented by 1.\n Please check channel:{sec_id}",
                                      UserWarning)

                # check data*.txt # no test.
                # check zero value in data, and interpolate. # check_zero
                chd = np.array(single_ch_data)
                if check_ch_zero:
                    zero_indices = np.where(chd == 0)[0]
                    if len(zero_indices) > 0:
                        warnings.warn(f"Zero values exist in channel.\n Please check channel:{sec_id}", UserWarning)
                        warnings.warn(f"data index:{zero_indices}", UserWarning)
                        if interpolate_ch_nearest:
                            clogger.info("Interpolating channel...")
                            single_ch_data = fill_zeros_with_nearest_value(chd)
                # append single channel and rf pulse data.
                channel_datas.append(single_ch_data)
                channel_rf_cnts.append(single_rf_cnt)

            # ----- Parse Trigger Data Blocks ----
            clogger.info( "STEP3.Parse Trigger Data Blocks")
            ## block header: the number of section. | 8 bytes
            trigger_num_sections = struct.unpack('<Q', fid.read(8))[0]  # 1
            log("Trigger_Num_Sections", trigger_num_sections)
            # clogger.info(fid.read(100))
            for tri_sec_id in range(1, trigger_num_sections + 1, 1):
                ## section header: the number of datas.
                ##  [data line number] * 9 [rf pulse(8 bytes),trigger_value(1 bytes)]
                # The number of bytes of the trigger | 13 bytes
                num_ch_bytes = struct.unpack('<Q', fid.read(8))[0]
                num_ch_line = int(num_ch_bytes / 13)
                log("Num_Ch_Bytes", num_ch_bytes)
                log("Num_Ch_Line", num_ch_line)
                single_trigger_cnt = []
                single_trigger_value = []
                ## decoding trigger data
                for i in range(num_ch_line):
                    ## trigger data definition
                    # sync pulse counting| 8 bytes | definition miss
                    sync_rf_id = struct.unpack('<2I', fid.read(8))
                    log("Sync_RF_id?", sync_rf_id)

                    # sync pulse counting| 8 bytes
                    sync_rf_count = struct.unpack('<I', fid.read(4))[0]
                    log("Sync_RF_Count", sync_rf_count)

                    # trigger value | the same as type.
                    sync_rf_value = struct.unpack('<B', fid.read(1))[0]
                    log("Sync_RF_Type", sync_rf_value)

                    single_trigger_cnt.append(sync_rf_count)
                    single_trigger_value.append(sync_rf_value)

                # append trigger info
                single_trigger_cnt = np.ceil(np.array(single_trigger_cnt) / 2)  # handle sampling point halving
                trigger_values.append(np.array(single_trigger_value))
                trigger_rf_cnts.append(single_trigger_cnt)

    except Exception as err:
        clogger.info(err)
    clogger.info("Mag file parsing is complete.")
    return {"patient_info": patient_info,
            "channel_datas": channel_datas, "channel_rf_cnts": channel_rf_cnts,
            "trigger_values": trigger_values, "trigger_rf_cnts": trigger_rf_cnts}


def _get_gender(code_byte):
    """get gender based on byte"""
    gender_code = {b'\x00': "Male",
                   b'\x01': "Female",
                   b'\x02': "Other"}
    gender_code = defaultdict(str, gender_code)
    return gender_code[code_byte]


def _get_filter_type(code_byte):
    """get the type of filter based on byte."""
    type_code = {
        b'\x00': "NoFilter",
        b'\x01': "LowPassFilter",
        b'\x02': "HighPassFilter",
        b'\x03': "BandPassFilter",
        b'\x04': "BandStopFilter",
        b'\x05': "NotchFilter",
    }
    type_code = defaultdict(str, type_code)
    return type_code[code_byte]



def opmag2fif(mag_path, fif_path, opm_position_path=None, ica_compatibility=True, check_rf_increment=True,
              check_ch_zero=True, interpolate_ch_nearest=True, verbose=False):
    """
    Convert Quan-Mag opm data(*.mag) to fif file(*.fif).
    Parameters
    ----------
    mag_path:str
        the path of quanmag opm-meg data(*.mag).
    fif_path:str
        the output path of converted file(*.fif).
    opm_position_path : str
        the path of opm sensor positions.(default:None)
    ica_compatibility: bool
        True:Used to plot_sensors()\ica.plot_properties in ica of MNE, But not compatible with Brainstorm. If you want to load in Brainstom, we should set the value of ica_compatibility to `False`.
    check_rf_increment : bool
        True indicates that the rf pulse need to be checked in increments of 1.
    check_ch_zero : bool
        True means check whether zero values exist in each channel.
    interpolate_ch_nearest: bool
        Notice,Prerequisite: work with `check_zero` parameter.
        If zero values are checked in channel data, interpolate with the nearest neighbor value.
    verbose:bool
        clogger.info log info.

    Returns
    -------
        dict of numpy object,including opm-meg data, trigger event et al.
    """
    raw_dict = _mag2data(mag_path=mag_path, check_rf_increment=check_rf_increment,
                         check_ch_zero=check_ch_zero, interpolate_ch_nearest=interpolate_ch_nearest, verbose=verbose)
    clogger.info("Start converting...")
    patient_info = raw_dict['patient_info']
    channel_datas = raw_dict['channel_datas']
    rf_pulse_cnts = raw_dict['channel_rf_cnts']
    trigger_values = raw_dict['trigger_values'][0]
    trigger_rf_cnts = raw_dict['trigger_rf_cnts']

    # get the minimum range of rf pulse cnt.
    rf_pulse_begin_cnt = []
    rf_pulse_end_cnt = []
    for rf in rf_pulse_cnts:
        if rf.size > 0:
            rf_pulse_begin_cnt.append(rf[0])
            rf_pulse_end_cnt.append(rf[-1])
    rf_pulse_begin_max_cnt = max(rf_pulse_begin_cnt)
    rf_pulse_end_min_cnt = min(rf_pulse_end_cnt)

    # extract data channels,and align channel data.
    crop_datas = []
    crop_min = 0
    crop_max = 0
    for ch_data, rf in zip(channel_datas, rf_pulse_cnts):
        if rf.size > 0:
            crop_min = int(np.where(rf_pulse_begin_max_cnt == rf)[0][0])
            crop_max = int(np.where(rf_pulse_end_min_cnt == rf)[0][0])
            crop_datas.append(ch_data[crop_min:crop_max])
        else:
            crop_datas.append(np.zeros((int(rf_pulse_end_min_cnt - rf_pulse_begin_max_cnt),)))

    crop_datas = np.array(crop_datas) * 1e-15  # convert fT to Telsa.

    # event trigger setting
    trigger_rf_cnts = trigger_rf_cnts - rf_pulse_begin_max_cnt
    trigger_chan = np.zeros((1, int(rf_pulse_end_min_cnt - rf_pulse_begin_max_cnt)))
    # set trigger value to trigger channel.
    trigger_index = np.array(trigger_rf_cnts[0, :][trigger_values != 0]).astype(int)
    trigger_chan[0, trigger_index] = trigger_values[trigger_values != 0]

    # combine trigger data.
    crop_datas = np.concatenate((crop_datas, trigger_chan), axis=0)

    # read opm positions
    if opm_position_path == None:
        opm_position_path = Path(__file__).parent / "data" / "opm_sanbo_3dmodel_positions64.txt"
    positions = pd.read_csv(opm_position_path, sep='\t')
    opm_64_ch_names = positions.loc[:, 'ch_name'].tolist()  # add stim channel, num: 65
    positions = positions.to_dict(orient='records')

    # format like Raw.Info['pos']
    for pos_row in positions:
        str_loc = pos_row['loc'].replace(' ', ',')
        pos_row['loc'] = np.array(eval(str_loc))

        if pos_row['kind'] == 1:
            pos_row['kind'] = FIFF.FIFFV_MEG_CH  # type: ignore
        elif pos_row['kind'] == 3:
            pos_row['kind'] = FIFF.FIFFV_STIM_CH  # type: ignore

        if pos_row['coil_type'] == 3024:
            pos_row['coil_type'] = FIFF.FIFFV_COIL_VV_MAG_T3  # type: ignore
        elif pos_row['coil_type'] == 0:
            pos_row['coil_type'] = FIFF.FIFFV_COIL_NONE  # type: ignore

        if pos_row['unit'] == 112:
            pos_row['unit'] = FIFF.FIFFB_CONTINUOUS_DATA  # type: ignore
        elif pos_row['unit'] == 107:
            pos_row['unit'] = FIFF.FIFFB_ISOTRAK  # type: ignore

    # Initialize an MNE info structure

    # # To plot ICA Components
    # # MEG/EOG/ECG sensors don't have digitization points; all requested
    # # channels must be EEG
    ch_types = ['mag'] * (len(opm_64_ch_names)-1)  # subtract stim channels.
    ch_types.append('stim')
    # opm_64_ch_names.append('STI101')

    # # set 64 OPM Layout
    # # While layouts are 2D locations, montages are 3D locations.
    # # Montages contain sensor positions in 3D (x, y, z in meters), which can be assigned to existing EEG/MEG data.
    # # If you’re working with EEG data exclusively, you’ll want to use Montages, not layouts.
    # # Layouts are idealized 2D representations of sensor positions.
    # # They are primarily used for arranging individual sensor subplots in a topoplot or for showing the approximate relative arrangement of sensors.
    # pos_dict = positions.set_index(0).apply(lambda x: x[[1, 2, 3]].values, axis=1).to_dict()
    #
    # opm_montage = mne.channels.make_dig_montage(pos_dict, coord_frame='head')

    info = mne.create_info(
        ch_names=opm_64_ch_names,
        ch_types=ch_types,
        sfreq=patient_info['sample_rate']  # clogger.info(raw.info['sfreq'])
    )

    # set other MNE Raw Info.
    info._unlocked = True
    info['chs'] = positions
    birthday = datetime.strptime(patient_info['patient_birthday'], '%Y-%m-%d')
    birthday = (int(birthday.year), int(birthday.month), int(birthday.day))
    gender = {"Other": 0, "Male": 1, "Female": 2}[patient_info['patient_gender']]
    info['subject_info'] = {'birthday': birthday, 'sex': gender,
                            'last_name': patient_info['patient_name'].encode("utf-8").decode("latin1")} # please use `.encode('latin1').decode('utf-8')` to decode Chinese content.
    info['experimenter'] = patient_info["Header"]
    info.set_meas_date(
        datetime.strptime(patient_info["exam_time"], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc))  # neet debug
    info._unlocked = False

    opm_raw = mne.io.RawArray(crop_datas, info)
    # opm_raw.set_montage(opm_montage)

    # change the type of channels to mag.
    # If you want to load in brainstorm, forbid the following code.
    # if ica_compatibility:
    #     ch_types = ['mag'] * (len(opm_64_ch_names) - 1)
    #     ch_types.append('stim')
    #     opm_raw.set_channel_types(mapping=dict(zip(opm_64_ch_names, ch_types)))

    if not exists(fif_path) and os.path.isdir(fif_path):
        os.makedirs(os.path.dirname(fif_path), exist_ok=True)

    # rm the same fif file
    if exists(fif_path):
        os.remove(fif_path)
    opm_raw.save(fif_path, overwrite=True, verbose=False)

    clogger.info("Convert OPM data to FIF file successfully.")
    clogger.info(f"Save FIF file to `{fif_path}` directory.")

def opmag2fif_cmd(mag_path, fif_path, opm_position_path=None, ica_compatibility=True, check_rf_increment=True,
                  check_ch_zero=True, interpolate_ch_nearest=True, verbose=False):
    opmag2fif(mag_path, fif_path, opm_position_path, ica_compatibility, check_rf_increment,
              check_ch_zero, interpolate_ch_nearest, verbose)


if __name__=="__main__":
    # just for test and debug.
    import time
    startt = time.time()
    opm_mag_dir = '/Users/reallo/Downloads/opm_artifacts/lp_opm_artifacts.mag'
    raw = opmag2fif(mag_path=opm_mag_dir, fif_path='/Users/reallo/Downloads/opm_artifacts/ta_raw.fif',ica_compatibility=False)
    endt = time.time()
    print(f"Time cost:{endt-startt}")

