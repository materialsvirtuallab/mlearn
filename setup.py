# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from setuptools import setup
from setuptools import find_packages

setup(
    name='mlearn',
    version='0.0.1',
    description='Benchmark suite for machine learning interatomic '
                'potentials for materials science.',
    download_url='https://github.com/materialsvirtuallab/mlearn.git',
    license='BSD',
    install_requires=['numpy', 'pandas', 'pymatgen', 'scikit-learn'],
    package=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)