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

from setuptools import setup, find_packages

VERSION = '0.1.2'
DISTNAME = "PyDemoTest"
DESCRIPTION = "TestDemo-Python project for uploading to pypi."
MAINTAINER = "reallo"
MAINTAINER_EMAIL = "liaopan_2015@163.com"
URL = "https://github.com/liaopan/"
LICENSE = "MIT-License"
DOWNLOAD_URL = "http://github.com/liaopan/**project"

with open("README.rst", "r") as fid:
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
    python_requires=">=3.6",
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
    keywords="Neuroscience neuroimaging MEG brain.",
    # project_urls={
    #     "Homepage": "",
    #     "Download": "",
    #     "Bug Tracker": "",
    #     "Forum": "",
    #     "Source Code": "",
    # },
    entry_points={
        "console_scripts": [
            "print_test = PyDemoTest:test_console_script",
        ]
    },
    # 表明当前模块依赖哪些包，若环境中没有，则会从pypi中自动下载安装！！！
    # install_requires=['docutils>=0.3'],
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

