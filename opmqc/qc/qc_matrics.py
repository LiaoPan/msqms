# -*- coding: utf-8 -*-
"""temporally record all quality control metric."""

"""
# 1. 实现柄哥常看的ICA图report.html（不是首要功能，仅需要评估数据质量）
    - 类似mne页面，点击切换？
        - UI界面，可以实时加载切换，查看质控情况？（要求依赖、编码难度稍高。）
        - html页面？把数据量大的结果放置在文件夹内，将简略信息，直接编码base64放置html内？| ICA成分多，生成缓慢。
        - 可以二次编程的嵌入jupyter的html页面？默认如果没有解释器，当成默认网页，如果存在解释器，则可以二次run，查看情况；
        
    - https://github.com/mne-tools/mne-icalabel 开源项目
# QC_Report.html
- SNR、NSR（noise-to-signal ratio）
- bad channel detection
    - ratio
- bad segments detection
    - 
- bad trails detection
    - 
- NaN value detection
- Infinite Value detection
- flat(or call constant) value detection
- high amplitude detection
- high frequency detection
- low correlation with other channels.?
    - neural signals recorded by neighboring electrodes are always similar.
- average and std of each channel by sensor type.


- the very fast parallel segment.
- the parallel filter.
"""
