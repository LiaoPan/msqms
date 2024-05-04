# -*- coding: utf-8 -*-
"""
Read BIDS datasets
"""
from mne.io import Raw
from mne_bids import BIDSPath, read_raw_bids
from mne_bids import print_dir_tree, make_report
from mne_bids import get_entity_vals
from typing import Literal, Optional, List


def read_raw_bids_dataset(bids_root: str, datatype: Literal['meg'], subjects: List[str] = None,
                          sessions: List[str] = None,
                          tasks: List[str] = None, suffixes: List[str] = None,
                          print_dir: bool = False, bids_report: bool = False) -> List[Raw]:
    """
    Read and load MEG data from a BIDS dataset.

    Parameters:
    bids_root (str): The root path of the BIDS dataset.
    datatype (Literal['meg']): The type of data to read, currently only 'meg' is supported.
    subjects (str, optional): The specific subjects to load. Default is None, which loads all subjects.
    sessions (str, optional): The specific sessions to load. Default is None, which loads all sessions.
    tasks (str, optional): The specific tasks to load. Default is None, which loads all tasks.
    suffixes (str, optional): Additional suffix to consider when loading data. Default is None.
    print_dir (bool, optional): If True, print the directory tree structure. Default is False.
    bids_report (bool, optional): If True, generate a BIDS report. Default is False.

    Returns:
    List[Raw]: A list of MNE Raw objects containing the loaded data.
    """

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

    # Load and read all raw data
    raw_list = []
    for subj in entities['subject']:
        for sess in entities['session']:
            for tk in entities['task']:
                if sess == '':
                    sess = None
                bids_path.update(subject=subj, session=sess, task=tk)
                raw = read_raw_bids(bids_path, verbose=False)
                raw_list.append(raw)
    return raw_list


if __name__ == "__main__":
    from mne.datasets import somato

    bids_root = somato.data_path()
    # print_dir_tree(bids_root,max_depth=3)
    # print("reports:",make_report(bids_root))
    datatype = 'meg'
    subject = ['01']
    task = ['somato']
    suffix = ['meg']
    session = ['']
    # raw_list = read_raw_bids_dataset(bids_root, datatype='meg', subjects=subject, sessions=session, tasks=task, suffixes=suffix,
    #                          print_dir=True)
    raw_lists = read_raw_bids_dataset(bids_root, datatype='meg', print_dir=True)
    print(raw_lists)
