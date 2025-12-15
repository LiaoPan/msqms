# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :
   Author :       LiaoPan
   date：         2023/8/28 15:00
   email:         liaopan_2015@163.com
   Copyright (C)    2023    Liao Pan
-------------------------------------------------
   Change Activity:
                   2023/8/28:
-------------------------------------------------
"""
__author__ = 'LiaoPan'
import os
import os.path as op
from setuptools import setup, find_packages

# get the version
version = None
with open(op.join('msqms', '_version.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

VERSION = version
DISTNAME = "msqms"
DESCRIPTION = "msqms is a fully automated quality control tool for OPM-MEG."
MAINTAINER = "reallo"
MAINTAINER_EMAIL = "liaopan_2015@163.com"
URL = "https://github.com/liaopan/msqms"
LICENSE = "MIT-License"
DOWNLOAD_URL = "http://github.com/liaopan/msqms"
REQUIREMENTS_PATH = "requirements.txt"
TEST_REQUIREMENTS_PATH = "requirements_testing_tools.txt"

def parse_requirements_file(fname: str) -> list:
    """
    Parameters
    ----------
    fname:str
        the path of requirements.txt

    Returns
    -------
    requirements: list
        list of requirements.
    """
    requirements = []
    if not os.path.exists(fname):
        return requirements

    with open(fname, "r") as fileid:
        for line in fileid:
            package_name = line.strip()
            if not package_name.startswith('#'):
                requirements.append(package_name)
    return requirements


install_requires = parse_requirements_file(REQUIREMENTS_PATH)
tests_requires = parse_requirements_file(REQUIREMENTS_PATH) + parse_requirements_file(TEST_REQUIREMENTS_PATH)

with open("README.md", "r", encoding="utf-8") as fid:
    long_description = fid.read()

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    download_url=DOWNLOAD_URL,
    packages=find_packages(),
    python_requires=">=3.8",
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
    ],
    version=VERSION,
    keywords="Neuroscience neuroimaging OPM-MEG and SQUID-MEG.",
    # project_urls={
    #     "Homepage": "",
    #     "Download": "",
    #     "Bug Tracker": "",
    #     "Forum": "",
    #     "Source Code": "",
    # },
    package_data={
        'msqms': ['conf/config.yaml', 'conf/opm/quality_config.yaml', 'conf/squid/quality_config.yaml',
                  "quality_reference/opm_quality_reference.yaml",
                  "quality_reference/squid_quality_reference.yaml",
                  "reports/templates/*","reports/templates/css/*","reports/templates/js/*"]
    },
    entry_points={
        "console_scripts": [
            "msqms_report = msqms.cli.workflow:generate_qc_report",
            "msqms_quality_ref_cal = msqms.cli.workflow:compute_and_update_quality_reference",
            "msqms_quality_ref_update = msqms.cli.workflow:update_quality_reference",
            "msqms_quality_ref_list = msqms.cli.workflow:list_quality_references"
        ]
    },
    # 表明当前模块依赖哪些包，若环境中没有，则会从pypi中自动下载安装！！！
    install_requires=install_requires,
    # 仅在测试时需要使用的依赖，在正常发布的代码中是没有用的。
    # 在执行python setup.py test时，可以自动安装这三个库，确保测试的正常运行。
    # tests_require=[
    #     'pytest>=3.3.1',
    #     'pytest-cov>=2.5.1',
    # ],
    # install_requires 在安装模块时会自动安装依赖包
    # 而 extras_require 不会，这里仅表示该模块会依赖这些包
    # 但是这些包通常不会使用到，只有当你深度使用模块时，才会用到，这里需要你手动安装
    # extras_require={
    #     'PDF':  ["ReportLab>=1.2", "RXP"],
    #     'reST': ["docutils>=0.3"],
    # }
)
