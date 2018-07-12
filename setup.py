# -*- coding: utf-8 -*-
#
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from setuptools import setup, find_packages

setup(
    name='pysteps',
    version='1.0',
    packages=find_packages(),
    license='LICENSE',
    description='Python framework for short-term ensemble prediction systems',
    long_description=open('README.rst').read(),
    classifiers=[
        'Development Status :: 5 - Production/Stable', 'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Cython'],    
)
