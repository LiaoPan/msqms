# -*- coding: utf-8 -*-
"""
MEG quality assessment based on MEG Signal Quality Metrics(MSQMs)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MSQM(object):
    def __init__(self,data):
        self.data = data

    def _get_paramters(self):
        pass

    def compute_msqm(self):
        raise NotImplementedError

