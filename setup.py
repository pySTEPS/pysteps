# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='pysteps',
    version='0.1',
    packages=find_packages(),
    license='LICENSE',
    include_package_data=True,
    description='Python framework for short-term ensemble prediction systems',
    long_description=open('README.rst').read(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3'],
    install_requires=['numpy', 'opencv-python', 'pillow', 'pyproj',
                      'attrdict', 'jsmin', 'scipy', 'matplotlib',
                      'jsonschema']
)
