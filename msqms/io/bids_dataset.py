# -*- coding: utf-8 -*-
"""
Read BIDS datasets
"""
from mne_bids import BIDSPath, read_raw_bids
from mne_bids import print_dir_tree, make_report
from mne_bids import get_entity_vals
from typing import Literal, Optional, List
from pathlib import Path
from tqdm.std import tqdm

def read_raw_bids_dataset(bids_root: Path, datatype: Literal['meg'] = 'meg', subjects: List[str] = None,
                          sessions: List[str] = None,
                          tasks: List[str] = None, runs: List[str] = None,
                          print_dir: bool = False, bids_report: bool = False) -> List:
    """
    Read and load MEG data from a BIDS dataset.

    Parameters
    ----------

    bids_root : str
        The root path of the BIDS dataset.
    datatype : {'meg'}
        The type of data to read, currently only 'meg' is supported.('eeg')
    subjects : str, optional
        The specific subjects to load. Default is None, which loads all subjects.
    sessions : str, optional
        The specific sessions to load. Default is None, which loads all sessions.
    tasks : str, optional
        The specific tasks to load. Default is None, which loads all tasks.
    runs : str, optional
        The specific runs to load. Default is None, which loads all runs.
    print_dir : bool, optional
        If True, print the directory tree structure. Default is False.
    bids_report : bool, optional
        If True, generate a BIDS report. Default is False.

    Returns
    -------
    list of str
        A list of MEG file paths.
    """
    bids_root = str(bids_root)

    if print_dir:
        print_dir_tree(bids_root, max_depth=3)
    if bids_report:
        print(make_report(bids_root))

    bids_path = BIDSPath(root=bids_root, datatype=datatype)
    entities = bids_path.entities

    for entity in bids_path.entities.keys():
        values = get_entity_vals(bids_root, entity, with_key=False)
        if values:
            entities[entity] = values  # get all subjects
        else:
            entities[entity] = ['']
    # Specify certain subjects, sessions, tasks if provided
    if subjects is not None:
        entities['subject'] = subjects
    if sessions is not None:
        entities['session'] = sessions
    if tasks is not None:
        entities['task'] = tasks
    if runs is not None:
        entities['run'] = runs

    # Load and read all raw data
    raw_list = []
    total_iters = len(entities['subject']) * len(entities['session']) * len(entities['task']) * len(entities['run'])
    with tqdm(total=total_iters) as pbar:
        for subj in entities['subject']:
            for sess in entities['session']:
                for tk in entities['task']:
                    if sess == '':
                        sess = None
                    for run in entities['run']:
                        try:
                            if run == '':
                                bids_path.update(subject=subj, session=sess, task=tk)
                            else:
                                bids_path.update(subject=subj, session=sess, task=tk, run=run)
                        except ValueError as e:
                            print("ValueError", e)
                            continue

                        try:
                            _ = read_raw_bids(bids_path, verbose=False)
                            # raw = mne.io.read_raw(bids_path,verbose=False)
                        except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
                            print("BIDS Parse Error:", e)
                            continue
                        raw_list.append(bids_path.copy())  # notice hell: 同一个内存地址
                        pbar.update(1)

    return raw_list


def get_subjects_from_bids(bids_root: Path):
    bids_path = BIDSPath(root=bids_root, datatype='meg')
    entities = bids_path.entities

    for entity in bids_path.entities.keys():
        values = get_entity_vals(bids_root, entity, with_key=False)
        if values:
            entities[entity] = values  # get all subjects
        else:
            entities[entity] = ['']

    return entities['subject']


def get_info_from_bids(bids_root: Path):
    bids_path = BIDSPath(root=bids_root, datatype='meg')
    entities = bids_path.entities

    for entity in bids_path.entities.keys():
        values = get_entity_vals(bids_root, entity, with_key=False)
        if values:
            entities[entity] = values  # get all subjects
        else:
            entities[entity] = ['']

    subjects = entities['subject']
    sessions = entities['session']
    tasks = entities['task']
    runs = entities['run']

    return (subjects, sessions, tasks, runs)


if __name__ == "__main__":
    from mne.datasets import somato

    bids_root = somato.data_path()
    # specified subject\task\session
    # datatype = 'meg'
    # subject = ['01']
    # task = ['somato']
    # suffix = ['meg']
    # session = ['']
    # raw_list = read_raw_bids_dataset(bids_root, datatype='meg', subjects=subject, sessions=session, tasks=task, suffixes=suffix,
    #                          print_dir=True)
    # example 1
    raw_lists = read_raw_bids_dataset(bids_root, datatype='meg', print_dir=True)
    print("raw_lists:", raw_lists)
    print("------")
    print(raw_lists[0])

    # example 2
    # raw_lists = read_raw_bids_dataset(r'C:\Data\Datasets\OPM-FACE.v2', datatype='meg', print_dir=True)
    # print(raw_lists)
